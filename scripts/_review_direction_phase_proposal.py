#!/usr/bin/env python3
"""생산코드를 바꾸지 않고 센서yaw 좌표변경에 공변인 평면각 식을 검산한다."""
from pathlib import Path
import time
import numpy as np
import _review_direction_metrics as base

OUT=base.OUT/'expanded46'


def phase_angle(rotation: np.ndarray) -> np.ndarray:
    """최근접 평면회전의 각도를 atan2(c-b,a+d)로 계산한다."""
    return np.arctan2(rotation[...,1,0]-rotation[...,0,1],rotation[...,0,0]+rotation[...,1,1])


def main() -> None:
    """모든 저장자세와 합성행렬에서 공변성 및 특이점 조건을 보고한다."""
    spec=base.read(OUT/'yaw_protocol.json');results={}
    for seq in spec['queries']:
        with np.load(base.cache_path(seq)) as cache:poses=cache['poses']
        relative=poses[0,:3,:3].T[None]@poses[:,:3,:3]
        original=phase_angle(relative)
        conditioning=np.hypot(relative[:,0,0]+relative[:,1,1],relative[:,1,0]-relative[:,0,1])
        valid=conditioning>1e-8
        errors={}
        for deg in spec['angles']:
            a=np.deg2rad(deg);c,s=np.cos(a),np.sin(a)
            r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            changed=phase_angle(relative@r.T)
            residual=np.angle(np.exp(1j*(changed-original+a)))
            errors[str(deg)]=float(abs(residual[valid]).max())
        results[seq]=dict(poses=len(poses),conditioning_min=float(conditioning.min()),near_singular_count=int((~valid).sum()),max_covariance_error_rad=errors)
    # 식의 성질은 회전행렬 여부보다2x2블록의 우측SO(2)변환에서 직접 성립한다.
    rng=np.random.default_rng(20261106);a=rng.normal(size=(10000,2,2))
    v=np.arctan2(a[:,1,0]-a[:,0,1],a[:,0,0]+a[:,1,1]);worst=0.
    for theta in [0.,.1,np.pi/2,np.pi,4.1]:
        r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        b=a@r.T;phi=np.arctan2(b[:,1,0]-b[:,0,1],b[:,0,0]+b[:,1,1])
        worst=max(worst,float(abs(np.angle(np.exp(1j*(phi-v+theta)))).max()))
    assert worst<1e-10
    for row in results.values():assert max(row['max_covariance_error_rad'].values())<1e-10
    base.write(OUT/'phase_covariance_proposal.json',dict(status='passed',created_unix=time.time(),formula='phi(R)=atan2(R10-R01,R00+R11)',
        identity='phi(R Rz(-theta))=phi(R)-theta mod2pi: complex number (R00+R11)+i(R10-R01) is multiplied by exp(-i theta).',
        singularity='두인수가모두0이면각미정. properSO3에서norm=1+R22이므로기울기180도에서특이하다. 근접특이에서수치조건악화.',
        scope='필요 수정의 수학적 후보만검사. 학습모델·원장·생산transport코드·원고는변경하지않음. 원래시점의Euler yaw와다르므로채택하려면학습/검증에서재학습·선택·정확도검증이필요. 시험성능을보고선택하지않음.',
        current='현재phi=atan2(R10,R00)는roll/pitch가있는자세에서일반적인우측센서yaw회전에공변하지않음.',
        synthetic_cases=50000,synthetic_max_error_rad=worst,sequences=results,source_sha256=base.digest(Path(__file__))))
    print('PHASE PROPOSAL PASSED',worst,flush=True)


if __name__=='__main__':main()
