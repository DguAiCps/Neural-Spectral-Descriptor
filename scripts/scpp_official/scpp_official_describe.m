function scpp_official_describe(work_dir, start_idx, lidar_height)
% Official Scan Context++ descriptors for one exported keyframe chunk.
%
% Calls the UNMODIFIED official T-RO functions of
% _external/scancontext_tro (commit b463bb7), tro2022/scancontext/desc/:
%   ptcloud2polarcontextAug.m  -> PC + 2 lateral root-shifted A-PC copies
%   ptcloud2cartcontext.m      -> CC
% The per-scan steps copy makeAugmentedExperienceAllFramesTreeMerge.m
% (tro2022/batch_processor/); differences are marked "DIFF:".
%
% In : {work_dir}/chunk_{start:06d}.xyz / .cnt (scripts/scpp_official/export_keyframes.py)
% Out: {work_dir}/desc_{start:06d}.mat with
%        pc_aug   20 x 60 x 3 x n  (:,:,1,i) = PC, (:,:,2/3,i) = root shift y -2 m / +2 m
%        cc       40 x 40 x n
%        n_points n x 1, t_pc / t_cc  n x 1 seconds
%
% lidar_height: the official constant is 1.9 m for every dataset
% (makeAugmentedExperienceAllFramesTreeMerge.m:6, makeExperience.m:4).

here = fileparts(mfilename('fullpath'));
tro = fullfile(here, '..', '..', '_external', 'scancontext_tro', 'tro2022');
addpath(fullfile(tro, 'scancontext', 'desc'));

% Parameters (paper Table II; batch scripts read them from SingleTestInfo res_/roi_)
NUM_RING = 20; NUM_SECTOR = 60; MAX_RANGE = 80;   % PC: res_ = [20 60], roi_(1) = 80
NUM_AUG_SCs = 3;                                  % ...AllFramesTreeMerge.m:18 (orig + 2)
CC_RES = [40 40]; CC_ROI = [100 40];              % CC: (5 m, 2 m) bins, x +-100 m, y +-40 m
axis_unit = [CC_ROI(1)*2/CC_RES(1), CC_ROI(2)*2/CC_RES(2)];  % ...AllFramesTreeMerge.m:128-131

base = fullfile(work_dir, sprintf('chunk_%06d', start_idx));
fid = fopen([base '.cnt'], 'r'); cnt = fread(fid, inf, 'int64=>double'); fclose(fid);
n = numel(cnt);
pc_aug = zeros(NUM_RING, NUM_SECTOR, NUM_AUG_SCs, n);
cc = zeros(CC_RES(1), CC_RES(2), n);
t_pc = zeros(n, 1); t_cc = zeros(n, 1);

fid = fopen([base '.xyz'], 'r');
for i = 1:n
    % DIFF: points come from our export (loader xyz as float32) instead of
    % readBin(scan_path); same 'single' -> double read as reader/readBin.m:4-5.
    points = fread(fid, [3, cnt(i)], 'single')';
    assert(size(points, 1) == cnt(i), 'short read in %s', base);

    points(:, 3) = points(:, 3) + lidar_height;   % ...AllFramesTreeMerge.m:112 / :123
    ptcloud = pointCloud(points);                 % :113 / :124

    tic;
    desc_list = ptcloud2polarcontextAug(ptcloud, NUM_SECTOR, NUM_RING, MAX_RANGE, NUM_AUG_SCs); % :119
    t_pc(i) = toc;
    for a = 1:NUM_AUG_SCs
        pc_aug(:, :, a, i) = desc_list{a};
    end

    % DIFF: the batch scripts compute either 'pc' or 'cc' per run
    % (testinfo.method_); here both are computed from the same cloud.
    tic;
    cc(:, :, i) = ptcloud2cartcontext(ptcloud, axis_unit, CC_ROI);   % :133
    t_cc(i) = toc;
end
fclose(fid);

n_points = cnt; %#ok<NASGU>
out = fullfile(work_dir, sprintf('desc_%06d.mat', start_idx));
tmp = [out(1:end-4) '_part.mat'];
save(tmp, 'pc_aug', 'cc', 'n_points', 't_pc', 't_cc', 'lidar_height', '-v7');
movefile(tmp, out);
fprintf('DESC %s n=%d  pc_aug %.1f ms/kf  cc %.1f ms/kf\n', out, n, ...
        1e3 * mean(t_pc), 1e3 * mean(t_cc));
end
