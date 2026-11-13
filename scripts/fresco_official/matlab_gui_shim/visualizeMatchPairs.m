function [] = visualizeMatchPairs(gt_poses, match_idx_pairs)
% 환경 shim(계산 무관): 공식 visualizeMatchPairs.m은 궤적 그림만 그린다(PR 카운트·기술자에 영향 없음).
% -nodisplay -batch에서 GUI 정지를 피하려고 그리지 않는다. 입력 크기만 기록한다.
fprintf('visualizeMatchPairs shim: %d poses, %d accepted pairs (not drawn)\n', size(gt_poses, 1), size(match_idx_pairs, 1));
end
