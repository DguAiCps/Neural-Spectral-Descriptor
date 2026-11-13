#!/usr/bin/env python3
"""정확한 격자 원형 이동의 보간·인코더 성질을 같은184스캔에서 확인한다."""
from pathlib import Path
import time
import numpy as np
import torch
import _review_direction_expand as expansion
from _review_direction_report import stats, unit
from encoding.range_image import interpolate_range_image

base, yaw = expansion.base, expansion.yaw
OUT = expansion.OUT / 'roll_check'


def main() -> None:
    """실제 원시 회전과 분리한 CPU 원리 검사를 수행한다."""
    grid = base.read(expansion.OUT/'grid_probe/protocol.json')
    protocol = dict(created_unix=time.time(), queries=grid['queries'], angles=grid['angles'],
        scope='기존격자검사의184원시스캔과3정수각도에서정확한격자roll을검사한다. 원시회전주평가가아닌CPU구현검산이며표본·각도를추가선택하지않는다.',
        sources={str(p.relative_to(base.ROOT)):base.digest(p) for p in [Path(__file__), expansion.OUT/'grid_probe/protocol.json',
            base.ROOT/'src/encoding/range_image.py', base.ROOT/'src/encoding/spectral_encoder.py']})
    base.write(OUT/'protocol.json', protocol)
    torch.set_num_threads(1)
    all_rows = []
    for sequence, queries in protocol['queries'].items():
        sensor, seq = sequence.split('/')
        loader = yaw._make_loader(sensor, seq)
        config = yaw.fixed_config() if sensor == 'nclt' else yaw._effective_config()
        encoder = yaw._make_desc_encoder(config, sensor, 'cpu').eval()
        with np.load(base.cache_path(sequence)) as archive:
            scan_ids = archive['scan_ids']
        rows = []
        for query in queries:
            points = loader[int(scan_ids[query])]['points']
            points = points[np.isfinite(points[:,:3]).all(1)]
            image = encoder.projector.project(points, keep_intensity=False)[0]
            original_image = interpolate_range_image(image, method='linear') if encoder.interpolate_empty else image
            with torch.no_grad():
                original = encoder.encode_projected_image(torch.from_numpy(original_image).float()).numpy()
            bev = yaw.BP.project(points, keep_intensity=False)[0]
            z0 = yaw._phase_sketch(yaw._pool_rows(np.maximum(bev, 0), 16), 8)
            for angle in protocol['angles']:
                shifted = np.roll(image, int(angle), axis=1)
                filled = interpolate_range_image(shifted, method='linear') if encoder.interpolate_empty else shifted
                with torch.no_grad():
                    encoded = encoder.encode_projected_image(torch.from_numpy(filled).float()).numpy()
                rolled_bev = np.roll(bev, int(angle/6), axis=1)
                z = yaw._phase_sketch(yaw._pool_rows(np.maximum(rolled_bev, 0), 16), 8)
                expected = z0*np.exp(-1j*np.deg2rad(angle)*np.arange(1,9))[None]
                rows.append(dict(sequence=sequence,query=query,angle=angle,
                    interpolation_commutation_max=float(abs(filled-np.roll(original_image,int(angle),axis=1)).max()),
                    range_descriptor_absolute_max=float(abs(encoded-original).max()),
                    range_descriptor_relative_l2=float(np.linalg.norm(encoded-original)/max(np.linalg.norm(original),1e-12)),
                    range_descriptor_unit_l2=float(np.linalg.norm(unit(encoded[None])-unit(original[None]))),
                    bev_phase_absolute_max=float(abs(z-expected).max()),
                    bev_phase_unit_l2=float(np.linalg.norm(unit(z[None])-unit(expected[None])))))
        base.write(OUT/(sequence.replace('/','_')+'.json'),dict(status='passed',rows=rows))
        all_rows += rows
        print('ROLL', sequence, len(rows), flush=True)
    fields = ['interpolation_commutation_max', 'range_descriptor_absolute_max', 'range_descriptor_relative_l2',
              'range_descriptor_unit_l2', 'bev_phase_absolute_max', 'bev_phase_unit_l2']
    result = {str(a):{f:stats(np.array([r[f] for r in all_rows if r['angle']==a])) for f in fields} for a in protocol['angles']}
    base.write(OUT/'summary.json',dict(status='passed',rows=len(all_rows),results=result,protocol_sha256=base.digest(OUT/'protocol.json')))
    print('ROLL COMPLETE', len(all_rows), flush=True)


if __name__ == '__main__':
    main()
