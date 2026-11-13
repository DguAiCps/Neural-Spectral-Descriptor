function pdv2_table3_scpp(action, n_map, repo)
% 공식 A-PC 실제 도착 계측. Python 래퍼가 같은 내용의 유효 함수명 파일을 만든다.
maxNumCompThreads(1);
work = fullfile(repo, 'results', 'pdv2_table3_20261029', sprintf('scpp_n%d', n_map));
write_json(fullfile(work, 'matlab_pid.json'), struct('pid', feature('getpid'), 'action', action));
tro = fullfile(repo, '_external', 'scancontext_tro', 'tro2022', 'scancontext');
addpath(fullfile(tro, 'desc'), fullfile(tro, 'alg'), fullfile(repo, 'scripts', 'scpp_official'));
parent = jsondecode(fileread(fullfile(repo, 'results', 'pdv2_table3_20261029', 'protocol.json')));
initial = n_map + 44;
ending = initial + parent.samples;

if strcmp(action, 'cache')
    started = tic;
    descriptor_file = fullfile(work, 'desc_000000.mat');
    if ~isfile(descriptor_file)
        scpp_official_describe(work, 0, 1.693);
    end
    reference_file = fullfile(work, 'official_reference.csv');
    if ~isfile(reference_file)
        scpp_official_match(work, ending, 'apc', 0, 30, reference_file);
    end
    mex('-R2018a', '-outdir', work, '-output', 'scpp_buffers_mex', ...
        fullfile(repo, 'scripts', '_pdv2_table3_scpp_buffers.c'));
    source = load(descriptor_file, 'pc_aug', 'lidar_height');
    assert(size(source.pc_aug, 4) == ending && source.lidar_height == 1.693);
    write_json(fullfile(work, 'cache_complete.json'), struct('frames', ending, ...
        'initial', initial, 'seconds_this_run', toc(started), 'matlab_version', version, ...
        'descriptor_reference', 'unchanged scpp_official_describe; A-PC 20x60x3', ...
        'matching_reference', 'unchanged scpp_official_match on full observed prefix'));
    return;
end
assert(strcmp(action, 'measure'));
assert(isfile(fullfile(work, 'cache_complete.json')));
addpath(work);
probe = ones(4,5); alias = probe; distinct = probe+1;
probe_addresses = scpp_buffers_mex({probe, alias, distinct});
assert(probe_addresses(1,1) == probe_addresses(2,1));
assert(probe_addresses(1,1) ~= probe_addresses(3,1));
assert(all(probe_addresses(:,2) == 160));
clear probe alias distinct probe_addresses;
source = load(fullfile(work, 'desc_000000.mat'), 'pc_aug', 'lidar_height');
reference = readmatrix(fullfile(work, 'official_reference.csv'), 'NumHeaderLines', 1);
assert(size(source.pc_aug, 4) == ending && source.lidar_height == 1.693);
fid = fopen(fullfile(work, 'chunk_000000.cnt'), 'r');
counts = fread(fid, Inf, 'int64=>double'); fclose(fid);
fid = fopen(fullfile(work, 'chunk_000000.xyz'), 'r');
fseek(fid, sum(counts(1:initial)) * 12, 'bof');
points = cell(parent.samples, 1);
for i = 1:parent.samples
    points{i} = fread(fid, [3, counts(initial+i)], 'single=>single')';
end
fclose(fid);

% 초기 지도 구축은 업데이트 비용과 분리한다. 배열 용량은 마지막 도착까지 고정한다.
started = tic;
D = zeros(20, 60, 3, ending);
RK = zeros(ending*3, 20);
SK = cell(ending*3, 1);
tree = [];
reset_state();
initial_build_s = toc(started);
checks = repmat(struct('arrival', 0, 'descriptor_max_error', 0, ...
    'winner', 0, 'score_error', 0), parent.warmup, 1);
for i = 1:parent.warmup
    arrival = initial + i - 1;
    desc = encode(points{i});
    err = max(abs(desc(:)-reshape(source.pc_aug(:,:,:,arrival+1), [], 1)));
    assert(err == 0, '원시 재생성 descriptor가 공식 참조와 다르다');
    append_state(desc, arrival+1);
    result = query(arrival-15);
    ref = reference(reference(:,1) == arrival-15, :);
    assert(size(ref,1) == 1 && result.winner == ref(7));
    assert(isequaln(result.confidence, -ref(6)));
    checks(i) = struct('arrival', arrival, 'descriptor_max_error', err, ...
        'winner', result.winner, 'score_error', abs(result.confidence+ref(6)));
end
verification = struct('status', 'passed', 'raw_regeneration', parent.warmup, ...
    'official_fullprefix_checks', checks, 'tolerance', ...
    'A-PC descriptor exact; official winner, augmentation, aligning shift and score exact; NaN preserved', ...
    'reference', 'official_reference.csv from unchanged scpp_official_match', ...
    'memory_alias_probe', 'same-storage alias and distinct-storage positive controls passed');

% 별도 준비와 검증이 끝난 뒤에만 순차 실제 도착 96개를 세 번 잰다.
rows = repmat(struct('repeat', 0, 'arrival', 0, 'query', 0, 'n_candidates', 0, ...
    'encode', 0, 'state_update', 0, 'update', 0, 'query_ms', 0, 'processing', 0, ...
    'winner', 0, 'confidence', 0, 'rerank_candidates', 1), parent.samples*parent.repeats, 1);
memories = cell(parent.repeats, 1);
for repeat = 1:parent.repeats
    reset_state();
    rss_before = process_memory();
    for i = 1:parent.samples
        arrival = initial+i-1;
        tick = tic; desc = encode(points{i}); encode_ms = 1e3*toc(tick);
        tick = tic; append_state(desc, arrival+1); state_ms = 1e3*toc(tick);
        tick = tic; result = query(arrival-15); query_ms = 1e3*toc(tick);
        % 참조 검사는 타이밍 바깥에서 모든 도착에 대해 한다.
        ref = reference(reference(:,1) == arrival-15, :);
        assert(result.winner == ref(7));
        assert(result.candidate_frame == ref(2) && result.candidate_aug == ref(3));
        assert(result.init_rot == ref(4) && result.argalign == ref(5));
        assert(isequaln(result.confidence, -ref(6)));
        rows((repeat-1)*parent.samples+i) = struct('repeat', repeat-1, ...
            'arrival', arrival, 'query', arrival-15, 'n_candidates', arrival-44, ...
            'encode', encode_ms, 'state_update', state_ms, 'update', encode_ms+state_ms, ...
            'query_ms', query_ms, 'processing', encode_ms+state_ms+query_ms, ...
            'winner', result.winner, 'confidence', result.confidence, 'rerank_candidates', 1);
    end
    % 마지막 단계의 전체 수치 버퍼를 조사하며 별도 원시/참조 fixture는 제외한다.
    memories{repeat} = state_memory(D, RK, SK, tree, ending, rss_before);
end
verification.measured_queries_checked = parent.samples*parent.repeats;
verification.tree_rows_final = (n_map+parent.samples-1)*3;
verification.memory_alias_audit = memories{end}.alias_audit;
result = struct('verification', verification, 'raw_samples', rows, ...
    'memory_repeats', [memories{:}], 'initial_cached_state_build_s', initial_build_s, ...
    'environment', struct('gpu', 'CPU only (GPU 0 bytes)', 'matlab', version, ...
        'platform', computer, 'threads', maxNumCompThreads, 'official', 'b463bb7 A-PC'));
write_json(fullfile(work, 'measurement.json'), result);
fprintf('SC++ COMPLETE N=%d samples=%d\n', n_map, numel(rows));

    function reset_state()
        D(:) = 0; RK(:) = 0; SK(:) = {[]}; tree = [];
        D(:,:,:,1:initial) = source.pc_aug(:,:,:,1:initial);
        for frame = 1:initial
            for augment = 1:3
                row = (frame-1)*3+augment;
                RK(row,:) = sc2invkey(D(:,:,augment,frame));
                SK{row} = sc2vkey(D(:,:,augment,frame));
            end
        end
    end

    function append_state(desc, frame)
        D(:,:,:,frame) = desc;
        for augment = 1:3
            row = (frame-1)*3+augment;
            RK(row,:) = sc2invkey(desc(:,:,augment));
            SK{row} = sc2vkey(desc(:,:,augment));
        end
    end

    function result = query(query_zero)
        query_one = query_zero+1;
        map_rows = (query_one-30)*3;
        query_row = (query_one-1)*3+1;
        tree = createns(RK(1:map_rows,:), 'NSMethod', 'kdtree');
        candidate = knnsearch(tree, RK(query_row,:), 'K', 1);
        frame = ceil(candidate/3);
        augment = candidate-(frame-1)*3;
        rotation = alignUsingSectorKey(SK{query_row}, SK{candidate});
        [distance, argalign] = sc_dist_fast_with_argalign(D(:,:,augment,frame), ...
            D(:,:,1,query_one), rotation, 0);
        nearest = 0; min_dist = inf;
        if distance < min_dist
            nearest = frame;
        end
        result = struct('winner', nearest-1, 'confidence', -distance, ...
            'candidate_frame', frame-1, 'candidate_aug', augment-1, ...
            'init_rot', rotation, 'argalign', argalign);
    end
end


function desc = encode(raw)
% 로더의 float32 입력 변환부터 공식 증강 표현까지 포함한다.
points = double(raw);
points(:,3) = points(:,3)+1.693;
cloud = pointCloud(points);
list = ptcloud2polarcontextAug(cloud, 60, 20, 80, 3);
desc = cat(3, list{:});
end


function result = state_memory(D, RK, SK, tree, count, rss_before)
% 객체를 재귀적으로 펼치되 배열은 참조로 전달하여 같은 주소를 한 번만 센다.
warning('off', 'MATLAB:structOnObject');
trees = struct(tree);
[arrays, labels] = flatten_numeric({D, RK, SK, trees}, 'state');
addresses = scpp_buffers_mex(arrays);
nonempty = addresses(:,2)>0;
addresses = addresses(nonempty,:); labels = labels(nonempty);
[unique_pointer, ~, groups] = unique(addresses(:,1));
sizes = zeros(numel(unique_pointer),1);
for i = 1:numel(unique_pointer)
    sizes(i) = double(max(addresses(groups==i,2)));
end
cpu = sum(sizes);
proc = process_memory();
leaves = cell(numel(labels),1);
for i = 1:numel(labels)
    leaves{i} = struct('path', labels{i}, 'bytes', double(addresses(i,2)), ...
        'pointer_hex', dec2hex(addresses(i,1)));
end
result = struct('cpu', cpu, 'gpu', 0, 'total', cpu, ...
    'model_gpu', 0, 'cuda_workspace_and_allocator_overhead', 0, ...
    'logical_feature_payload_bytes_per_frame', 20*60*3*8, ...
    'logical_feature_payload_bytes', count*20*60*3*8, ...
    'timepoint', 'after last measured arrival', ...
    'process_rss_before', rss_before.rss, 'process_rss_after', proc.rss, ...
    'process_peak_rss', proc.peak, 'cuda_allocated', 0, 'cuda_reserved', 0, ...
    'cuda_peak_allocated', 0, 'cuda_peak_reserved', 0, ...
    'breakdown', struct('D', numel(D)*8, 'RK', numel(RK)*8, ...
        'SK', sum(cellfun(@numel, SK))*8, 'tree_unique_and_metadata_numeric', ...
        cpu-numel(D)*8-numel(RK)*8-sum(cellfun(@numel, SK))*8), ...
    'alias_audit', struct('numeric_leaves', size(addresses,1), ...
        'unique_buffers', numel(unique_pointer), 'duplicate_aliases', size(addresses,1)-numel(unique_pointer), ...
        'sum_before_dedup', sum(double(addresses(:,2))), 'unique_bytes', cpu, ...
        'tree_class', class(tree), 'tree_fields', {fieldnames(trees)}, 'buffer_leaves', {leaves}), ...
    'capacity_note', 'D/RK fixed exact final observation capacity; SK includes all observed descriptors; tree includes exact final eligible prefix and its own internal numeric buffers. MATLAB cell/object headers, interpreter and input/audit fixtures appear in RSS only.');
end


function [arrays, labels] = flatten_numeric(value, label)
arrays = {}; labels = {};
if isnumeric(value) || islogical(value) || ischar(value)
    arrays = {value}; labels = {label};
elseif iscell(value)
    for i = 1:numel(value)
        [a,l] = flatten_numeric(value{i}, sprintf('%s{%d}',label,i));
        arrays = [arrays, a]; labels = [labels,l]; %#ok<AGROW>
    end
elseif isstruct(value)
    fields = fieldnames(value);
    for i = 1:numel(value)
        for j = 1:numel(fields)
            [a,l] = flatten_numeric(value(i).(fields{j}), sprintf('%s(%d).%s',label,i,fields{j}));
            arrays = [arrays,a]; labels = [labels,l]; %#ok<AGROW>
        end
    end
elseif isobject(value)
    [arrays,labels] = flatten_numeric(struct(value), [label,'.object']);
elseif ~isempty(value)
    error('지원하지 않은 상태 메모리 종류: %s',class(value));
end
end


function result = process_memory()
text = fileread('/proc/self/status');
rss = regexp(text, 'VmRSS:\s+(\d+)\s+kB', 'tokens', 'once');
peak = regexp(text, 'VmHWM:\s+(\d+)\s+kB', 'tokens', 'once');
result = struct('rss', str2double(rss{1})*1024, 'peak', str2double(peak{1})*1024);
end


function write_json(path, value)
temporary = [path,'.part'];
fid = fopen(temporary, 'w');
assert(fid >= 0);
fprintf(fid, '%s\n', jsonencode(value)); fclose(fid);
movefile(temporary, path);
end
