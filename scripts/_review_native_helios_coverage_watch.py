#!/usr/bin/env python3
"""새 공개 입력의 독립 검산이 끝날 때 범위 보고서를 갱신한다."""
import json
import os
import subprocess
import time

import _review_native_helios as native
from _review_native_helios_regen_eval import OUT as REGEN


def main() -> None:
    """완료된 검산 파일의 변경만 감시하며 GPU 실행에는 관여하지 않는다."""
    previous = None
    while True:
        state = {"pid": os.getpid(), "unix_time": time.time()}
        try:
            signature = [native.sha(native.OUT.parent / "learned_independent_verification.json"), native.sha(REGEN / "independent_verification.json")]
            if signature != previous:
                subprocess.run(["python3", "scripts/_review_native_helios_coverage.py"], cwd=native.REPO, check=True)
                previous = signature
            report = json.loads((native.OUT / "coverage_with_regeneration.json").read_text())
            state.update(available=report["available_native_pairs"], required=24, released_pairs=report["released_input_pairs"], regenerated_pairs=report["new_regenerated_input_pairs"], passed=True)
        except Exception as exc:
            state.update(passed=False, error=repr(exc))
        native.write(native.OUT / "coverage_watch_status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
        if state.get("released_pairs") == 24:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
