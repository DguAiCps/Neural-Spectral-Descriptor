function review_remaining_native_scpp(repo, limit)
% 공식 MATLAB PC/CC의 KITTI00 대표 재현 진단. 로컬 실행은 사용자 승인 후.
% matlab -singleCompThread -nojvm -batch "addpath('scripts'); review_remaining_native_scpp(pwd,12)"
% limit=12: 메모리/실행시간 사전 확인. limit=4541: 전체 시퀀스.
tro = fullfile(repo, '_external', 'scancontext_tro', 'tro2022');
addpath(fullfile(tro, 'reader'), fullfile(tro, 'scancontext', 'desc'), ...
        fullfile(tro, 'scancontext', 'alg'));
out = fullfile(repo, 'results', 'pdv2_remaining_native_20261106', 'scpp');
if ~exist(out, 'dir'), mkdir(out); end
data = '/mnt/d/NSD_datasets/kitti/dataset';
poses = readmatrix(fullfile(data, 'poses', '00.txt'));
xyz = poses(:, [4, 8, 12]);
n = min(limit, size(poses, 1));
pc = zeros(20, 60, n); cc = zeros(40, 40, n);
cache = fullfile(out, sprintf('descriptors_%d.mat', n));
t = tic;
if exist(cache, 'file')
    load(cache, 'pc', 'cc');
else
    for i = 1:n
        raw = fullfile(data, 'sequences', '00', 'velodyne', sprintf('%06d.bin', i-1));
        points = readBin(raw);
        points(:, 3) = points(:, 3) + 1.9;
        cloud = pointCloud(points);
        pc(:, :, i) = ptcloud2polarcontext(cloud, 60, 20, 80);
        cc(:, :, i) = ptcloud2cartcontext(cloud, [5, 2], [100, 40]);
        if mod(i, 100) == 0 || i == n
            fprintf('descriptors %d/%d elapsed %.1fs\n', i, n, toc(t));
        end
    end
    save(cache, 'pc', 'cc', '-v7');
end
descriptor_s = toc(t);
if n < 4541
    assert(all(isfinite(pc(:))) && all(isfinite(cc(:))));
    report = struct('frames', n, 'descriptor_s', descriptor_s, ...
                    'descriptor_bytes', 8 * (numel(pc) + numel(cc)), ...
                    'scope', '소규모 공식 MATLAB 실행 확인; 논문 수치 재현 아님');
    fid = fopen(fullfile(out, 'smoke.json'), 'w');
    fprintf(fid, '%s\n', jsonencode(report)); fclose(fid);
    return;
end
% 논문은 시간 제외 길이를 명시하지 않는다. 공개 평가기의 최근49개 제외를
% 고정 진단으로 사용한다. 논문 표III 재방문852개와의 일치는 따로 확인한다.
gap = 49;
ledger = struct();
for variant = {'pc', 'cc'}
    name = variant{1};
    if strcmp(name, 'pc'), desc = pc; else, desc = cc; end
    key = zeros(n, size(desc, 1));
    sector = cell(n, 1);
    for i = 1:n
        key(i, :) = sc2invkey(desc(:, :, i));
        sector{i} = sc2vkey(desc(:, :, i));
    end
    result = zeros(n-gap, 5);
    for q = gap+1:n
        last = q-gap;
        tree = createns(key(1:last, :), 'NSMethod', 'kdtree');
        winner = knnsearch(tree, key(q, :), 'K', 1);
        alignment = alignUsingSectorKey(sector{q}, sector{winner});
        distance = sc_dist_fast_with_argalign(desc(:, :, winner), desc(:, :, q), alignment, 0);
        geo = sqrt(sum((xyz(1:last, [1, 3])-xyz(q, [1, 3])).^2, 2));
        result(q-gap, :) = [q-1, winner-1, distance, geo(winner), min(geo)];
    end
    ledger.(name) = result;
    writematrix(result, fullfile(out, [name '_ledger.csv']));
end
report = struct('frames', n, 'gap', gap, 'radius_m', 8, 'k', 1, ...
                'descriptor_s', descriptor_s, 'elapsed_s', toc(t), ...
                'scope', '논문 PC/CC 연산·전체 KITTI00; 정확한 시간 제외/질의 목록 미공개로 조건부 진단');
fid = fopen(fullfile(out, 'execution.json'), 'w');
fprintf(fid, '%s\n', jsonencode(report)); fclose(fid);
end
