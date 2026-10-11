function scpp_official_match(work_dir, n_frames, variant, search_ratio, min_gap, out_csv)
% Online past-only Scan Context++ place recognition with the official
% retrieval flow, over the descriptors of scpp_official_describe.m.
%
% Per query t (0-based, t >= min_gap) the map is every keyframe j <= t - min_gap:
%   1. retrieval key (sc2invkey) k-d tree over the map -- one tree holding
%      the original AND augmented keys of each map place (paper Sec. V-D);
%      rebuilt for every query over the exact prefix, as the official
%      evaluator does (fast_evaluator/main.m:75) -- k = 1 (paper Sec. V-A,
%      Table II, Sec. VIII-B);
%   2. pre-alignment with the aligning key: alignUsingSectorKey(sc2vkey)
%      (paper eq. 4);
%   3. full-descriptor distance in the neighbourhood of that shift:
%      sc_dist_fast_with_argalign(candidate, query, init_rot, search_ratio)
%      (paper eq. 5; search_ratio = 0 is the paper's N(n*) = {n*});
%   4. the official candidate loop (fast_evaluator/main.m:85-97): the
%      candidate becomes the match only if dist < inf (a NaN distance --
%      no column with points in both -- leaves no match).
% Official functions used unchanged: sc2invkey, sc2vkey, alignUsingSectorKey,
% sc_dist_fast_with_argalign (tro2022/scancontext/), createns / knnsearch.
%
% variant: 'pc'  polar context, original only
%          'apc' polar context + 2 lateral root-shift augmentations (A-PC)
%          'cc'  Cartesian context, original only
%          'acc' Cartesian context + double flip (A-CC; paper Sec. V-D.2 --
%                the flip itself is NOT in the released code, see DIFF below)
% Output CSV, one row per query:
%   query_idx, cand_frame, cand_aug, init_rot, argalign, cand_dist, j_star, n_tree
% (0-based frame / aug indices; j_star = -1 and the ledger score NaN when the
%  official loop accepts no candidate.)

here = fileparts(mfilename('fullpath'));
tro = fullfile(here, '..', '..', '_external', 'scancontext_tro', 'tro2022');
addpath(fullfile(tro, 'scancontext', 'desc'));
addpath(fullfile(tro, 'scancontext', 'alg'));

% ---- load descriptors (chunks in index order)
files = dir(fullfile(work_dir, 'desc_*.mat'));
files = files(~contains({files.name}, '_part'));
[~, order] = sort({files.name});
files = files(order);
D = []; t_pc = []; t_cc = []; n_points = []; lh = [];
for k = 1:numel(files)
    s = load(fullfile(work_dir, files(k).name));
    t_pc = [t_pc; s.t_pc]; t_cc = [t_cc; s.t_cc]; n_points = [n_points; s.n_points]; %#ok<AGROW>
    lh = unique([lh, s.lidar_height]);
    switch variant
        case 'pc',  d = s.pc_aug(:, :, 1, :);
        case 'apc', d = s.pc_aug;
        case 'cc',  d = reshape(s.cc, size(s.cc, 1), size(s.cc, 2), 1, []);
        case 'acc'
            c = reshape(s.cc, size(s.cc, 1), size(s.cc, 2), 1, []);
            % DIFF: A-CC "double flip on both axes" (paper Sec. V-D.2); the
            % released batch scripts support augmentation for 'pc' only.
            d = cat(3, c, flip(flip(c, 1), 2));
        otherwise, error('unknown variant %s', variant);
    end
    D = cat(4, D, d);
end
assert(size(D, 4) == n_frames, 'descriptor count %d != n_frames %d', size(D, 4), n_frames);
assert(isscalar(lh), 'chunks were described with different lidar_height');
n_aug = size(D, 3);
n_ring = size(D, 1);

% ---- sub-descriptors, row r = (frame - 1) * n_aug + aug
RK = zeros(n_frames * n_aug, n_ring);
SK = cell(n_frames * n_aug, 1);
for i = 1:n_frames
    for a = 1:n_aug
        r = (i - 1) * n_aug + a;
        RK(r, :) = sc2invkey(D(:, :, a, i));
        SK{r} = sc2vkey(D(:, :, a, i));
    end
end

% ---- online loop (1-based query T = t + 1)
n_q = n_frames - min_gap;
rows = nan(max(n_q, 0), 8);
tic;
for T = (min_gap + 1):n_frames
    m = (T - min_gap) * n_aug;              % map = frames 1..T-min_gap, every aug
    q = (T - 1) * n_aug + 1;                % query = original descriptor of frame T
    tree = createns(RK(1:m, :), 'NSMethod', 'kdtree');
    cand = knnsearch(tree, RK(q, :), 'K', 1);

    f = ceil(cand / n_aug);
    a = cand - (f - 1) * n_aug;
    init_rot = alignUsingSectorKey(SK{q}, SK{cand});
    % alignUsingSectorKey shifts the CANDIDATE key, and sc_dist_fast*
    % shifts its FIRST argument, so the candidate goes first.
    [dist, argalign] = sc_dist_fast_with_argalign(D(:, :, a, f), D(:, :, 1, T), init_rot, search_ratio);

    nearest_idx = 0; min_dist = inf;        % fast_evaluator/main.m:85-97, K = 1
    if dist < min_dist
        nearest_idx = f; min_dist = dist; %#ok<NASGU>
    end
    rows(T - min_gap, :) = [T - 1, f - 1, a - 1, init_rot, argalign, dist, nearest_idx - 1, m];
end
t_loop = toc;

fid = fopen([out_csv '.part'], 'w');
fprintf(fid, 'query_idx,cand_frame,cand_aug,init_rot,argalign,cand_dist,j_star,n_tree\n');
fprintf(fid, '%d,%d,%d,%d,%d,%.17g,%d,%d\n', rows.');
fclose(fid);
movefile([out_csv '.part'], out_csv);
side = struct('variant', variant, 'search_ratio', search_ratio, 'lidar_height', lh, ...
              'min_gap', min_gap, 'n_aug', n_aug, 'matlab_version', version, ...
              'desc_pc_aug_ms_per_kf', 1e3 * mean(t_pc), 'desc_cc_ms_per_kf', 1e3 * mean(t_cc), ...
              'match_ms_per_query', 1e3 * t_loop / max(n_q, 1), 'n_points_mean', mean(n_points));
fid = fopen([out_csv '.json'], 'w'); fprintf(fid, '%s', jsonencode(side)); fclose(fid);
fprintf('MATCH %s variant=%s n_q=%d no_match=%d  %.2f ms/query\n', out_csv, variant, ...
        n_q, sum(rows(:, 7) < 0), 1e3 * t_loop / max(n_q, 1));
end
