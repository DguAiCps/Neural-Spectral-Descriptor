#!/usr/bin/env python3
"""정수bin 원시회전에서 점의 bin·범위·셀집계가 바뀌는 원인을 확인한다."""
from pathlib import Path
import time
import numpy as np
import torch
import _review_direction_expand as expansion
from encoding.range_image import interpolate_range_image

base,yaw=expansion.base,expansion.yaw
OUT=expansion.OUT/'grid_probe'
ANGLES=(6.,90.,180.)


def assignments(points, projector, bev=False):
    """원래 투영기와 같은 자료형·floor·범위조건을 점별로 재현한다."""
    x,y,z=points[:,:3].T
    theta=(np.arctan2(y,x)+np.pi)%(2*np.pi)
    n=projector.n_sectors if bev else projector.n_azimuth
    col=np.clip(np.floor(theta/(2*np.pi)*n).astype(int),0,n-1)
    if bev:
        r=np.sqrt(x**2+y**2)
        valid=(r>=projector.min_range)&(r<projector.max_range)&(z>projector.z_min)
        row=np.clip(np.floor(r-projector.min_range).astype(int),0,projector.n_rings-1)
        value=z
    else:
        xx,yy,zz=[np.clip(v**2,0,1e10) for v in (x,y,z)]
        r=np.sqrt(xx+yy+zz)
        elevation=np.arctan2(z,np.sqrt(xx+yy))
        row=np.clip(np.floor((elevation-projector.elevation_min)/(projector.elevation_max-projector.elevation_min)*projector.n_elevation).astype(int),0,projector.n_elevation-1)
        valid=(r>=projector.min_range)&(r<=projector.max_range)&np.isfinite(r)
        value=r
    return dict(row=row,col=col,valid=valid,value=value)


def grid(assign, shape, bev=False):
    """점별 bin을 min거리 또는 max높이로 집계한다."""
    flat=np.full(np.prod(shape),-np.inf if bev else np.inf,np.float32)
    keep=assign['valid'];ids=assign['row'][keep]*shape[1]+assign['col'][keep]
    operation=np.maximum.at if bev else np.minimum.at
    operation(flat,ids,assign['value'][keep].astype(np.float32))
    flat[~np.isfinite(flat)]=0
    return flat.reshape(shape)


def error(a,b):
    """격자·특징 차이를 유한한 수로 요약한다."""
    return dict(max_absolute=float(np.max(abs(a-b))),l2=float(np.linalg.norm(a-b)),changed_cells=int(np.sum(abs(a-b)>1e-5)))


def main():
    """46시퀀스 각각 고정4질의의 원시 점 할당과 내부대조를 검사한다."""
    spec=expansion.freeze();OUT.mkdir(exist_ok=True)
    plan=dict(created_unix=time.time(),queries={s:q[::16] for s,q in spec['queries'].items()},angles=ANGLES,
        scope='각 시퀀스 고정64질의 중0,16,32,48번째 중심184스캔. 원시점 float32회전, float64진단, 각도bin만 고정, 모든bin·범위 고정 대조. 최종 장소인식 평가가 아니라 floor와셀집계의 원인검사.',
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in [Path(__file__),base.ROOT/'src/encoding/range_image.py',base.ROOT/'src/encoding/bev_image.py',expansion.OUT/'yaw_protocol.json']})
    base.write(OUT/'protocol.json',plan)
    torch.set_num_threads(1)
    for sequence,queries in plan['queries'].items():
        name=sequence.replace('/','_');target=OUT/(name+'.json')
        if target.exists():continue
        sensor,seq=sequence.split('/')
        loader=yaw._make_loader(sensor,seq)
        config=yaw.fixed_config() if sensor=='nclt' else yaw._effective_config()
        encoder=yaw._make_desc_encoder(config,sensor,'cpu').eval()
        cache=dict(np.load(base.cache_path(sequence)))
        rows=[]
        for q in queries:
            points=loader[int(cache['scan_ids'][q])]['points']
            points=points[np.isfinite(points[:,:3]).all(1)]
            for angle in ANGLES:
                r=yaw.rotz(angle)
                rotated=points.copy();rotated[:,:3]=points[:,:3].astype(np.float64)@r.T
                p64=points.astype(np.float64);r64=p64.copy();r64[:,:3]=p64[:,:3]@r.T
                for label,projector,bev in [('range',encoder.projector,False),('bev',yaw.BP,True)]:
                    original=assignments(points,projector,bev);actual=assignments(rotated,projector,bev)
                    original64=assignments(p64,projector,bev);actual64=assignments(r64,projector,bev)
                    cols=projector.n_sectors if bev else projector.n_azimuth
                    shape=(projector.n_rings if bev else projector.n_elevation,cols)
                    shift=int(round(angle*cols/360))
                    expected=(original['col']+shift)%cols
                    locked_angle=dict(actual,col=expected)
                    locked_all=dict(actual,col=expected,row=original['row'],valid=original['valid'])
                    g0,ga,gang,gall=[grid(a,shape,bev) for a in (original,actual,locked_angle,locked_all)]
                    np.testing.assert_array_equal(g0,projector.project(points,keep_intensity=False)[0])
                    np.testing.assert_array_equal(ga,projector.project(rotated,keep_intensity=False)[0])
                    ideal=np.roll(g0,shift,axis=1)
                    common=original['valid']&actual['valid'];common64=original64['valid']&actual64['valid']
                    detail=dict(sequence=sequence,query=int(q),angle=angle,projector=label,points=len(points),valid_points=int(common.sum()),
                        angular_assignment_mismatch=int((common&(actual['col']!=expected)).sum()),
                        angular_assignment_mismatch_float64=int((common64&(actual64['col']!=(original64['col']+shift)%cols)).sum()),
                        row_assignment_mismatch=int((common&(actual['row']!=original['row'])).sum()),
                        range_mask_mismatch=int((original['valid']!=actual['valid']).sum()),
                        actual_grid_vs_roll=error(ga,ideal),angle_locked_vs_roll=error(gang,ideal),all_bins_locked_vs_roll=error(gall,ideal))
                    if bev:
                        coeff=[yaw._phase_sketch(yaw._pool_rows(np.maximum(g,0),16),8) for g in (ideal,ga,gang,gall)]
                        detail['coefficient_errors']={k:error(v,coeff[0]) for k,v in zip(('actual','angle_locked','all_bins_locked'),coeff[1:])}
                    else:
                        values=[]
                        for g in (ideal,ga,gang,gall):
                            if encoder.interpolate_empty:g=interpolate_range_image(g,method='linear')
                            with torch.no_grad():values.append(encoder.encode_projected_image(torch.from_numpy(g).float()).numpy())
                        detail['descriptor_errors']={k:error(v,values[0]) for k,v in zip(('actual','angle_locked','all_bins_locked'),values[1:])}
                    rows.append(detail)
        base.write(target,dict(status='passed',rows=rows,protocol_sha256=base.digest(OUT/'protocol.json')))
        print('GRID',sequence,len(rows),flush=True)


if __name__=='__main__':main()
