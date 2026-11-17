#!/usr/bin/env bash
# 본 비용 감독이 끝나고 "all complete"일 때만 부록 지도크기 감독을 이어서 실행한다(GPU 동시 실행 방지).
REPO=/home/pkm/Neural-Spectral-Codec
WORK=$REPO/results/pdv2_baseline_revision_20261107/ringpp
MAIN_PID=$(cat "$WORK/supervisor.pid")
echo $$ > "$WORK/chain.pid"
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 30; done
if python3 -c "import json,sys; s=json.load(open('$WORK/supervisor_status.json')); sys.exit(0 if (s['stage'],s['state'])==('all','complete') else 1)"; then
  echo "$(date -Is) main complete; starting mapsize" >> "$WORK/chain.log"
  cd "$REPO" && exec scripts/_ringpp_exactmax_supervisor_mapsize_20261107.sh
else
  echo "$(date -Is) main not complete; mapsize not started" >> "$WORK/chain.log"
fi
