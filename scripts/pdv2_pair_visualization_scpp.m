function pdv2_pair_visualization_scpp(repo, name, output_root)
% 고정 쌍의 A-PC 매칭. 원래 K=1 후보 검색 대신 주어진 참조를 비교한다.
maxNumCompThreads(1);
if nargin < 3
    output_root = 'pdv2_pair_visualization_20261030';
end
work = fullfile(repo, 'results', output_root, name);
addpath(fullfile(repo, 'scripts', 'scpp_official'));
addpath(fullfile(repo, '_external', 'scancontext_tro', 'tro2022', 'scancontext', 'desc'));
addpath(fullfile(repo, '_external', 'scancontext_tro', 'tro2022', 'scancontext', 'alg'));
assert(isfile(fullfile(work, 'raw_complete.json')));
spec = jsondecode(fileread(fullfile(work, 'scpp_input.json')));
if ~isfile(fullfile(work, 'desc_000000.mat'))
    scpp_official_describe(work, 0, spec.lidar_height);
end
source = load(fullfile(work, 'desc_000000.mat'), 'pc_aug');
D = source.pc_aug;
assert(size(D,4) == numel(spec.indices));
RK = zeros(size(D,4), 3, 20);
SK = cell(size(D,4), 3);
for frame = 1:size(D,4)
    for a = 1:3
        RK(frame,a,:) = sc2invkey(D(:,:,a,frame));
        SK{frame,a} = sc2vkey(D(:,:,a,frame));
    end
end
rows = readmatrix(fullfile(work, 'scpp_pairs.csv'), 'NumHeaderLines', 1);
scores = zeros(size(rows,1), 5);
for n = 1:size(rows,1)
    q = rows(n,3)+1; j = rows(n,4)+1;
    differences = reshape(RK(j,:,:),3,20)-reshape(RK(q,1,:),1,20);
    [~, augmentation] = min(sum(differences.^2,2));
    shift = alignUsingSectorKey(SK{q,1}, SK{j,augmentation});
    distance = sc_dist_fast_with_argalign(D(:,:,augmentation,j),D(:,:,1,q),shift,0);
    scores(n,:) = [rows(n,1:2), -distance, augmentation-1, shift];
end
fid = fopen(fullfile(work, 'scpp_scores.csv'), 'w');
fprintf(fid,'query,reference,score,augmentation,shift\n');
fprintf(fid,'%d,%d,%.17g,%d,%d\n',scores');
fclose(fid);
fprintf('PAIR SC++ COMPLETE %s %d pairs\n', name, size(rows,1));
end
