#!/usr/bin/env python3
"""회전이 완료된 시퀀스부터 원인대조를 실행해 원시 재생과 겹쳐 처리한다."""
import time
import torch
import _review_direction_causes as causes


def main() -> None:
    """동일 실험 조건을 유지하고 준비된 입력의 실행 순서만 조정한다."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    spec=causes.freeze()
    pending=list(spec['queries'])
    while pending:
        ready=[s for s in pending if (causes.expansion.OUT/'yaw'/s.replace('/','_')/'rotate.json').exists()]
        if not ready:
            time.sleep(5)
            continue
        sequence=ready[0]
        causes.run_sequence(sequence)
        pending.remove(sequence)
        print('QUEUE',46-len(pending),46,flush=True)
    print('ALL CAUSES COMPLETE',flush=True)


if __name__=='__main__':main()
