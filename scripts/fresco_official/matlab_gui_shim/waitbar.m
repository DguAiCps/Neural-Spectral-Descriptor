function h = waitbar(x, varargin)
% 환경 shim(계산 무관): 공식 createDescsAndRetrieve.m의 진행 표시줄. -nodisplay -batch에서 원래 waitbar가
% 'Computing similarities' 단계에서 멈춰(CPU 증가 없음, 2026-11-07 기록) 보이지 않는 figure로 대체한다.
% 공식 코드가 쓰는 get(findobj(h,'Type','axes'),'Title'), set(..), delete(h)가 그대로 동작한다.
persistent fx_h
if nargin >= 2 && ~ischar(varargin{1}) && ~isstring(varargin{1})
    h = varargin{1};     % waitbar(x, h, msg): 갱신은 생략
    return;
end
fx_h = figure('Visible', 'off');
axes('Parent', fx_h);
h = fx_h;
end
