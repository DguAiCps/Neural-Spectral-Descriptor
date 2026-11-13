#!/usr/bin/env python3
"""검증 지표와 KITTI 손실 분해를 겹침 없이 표시한다."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """고정한 통합 결과에서 PNG와 PDF를 만든다."""
    root = Path(__file__).resolve().parents[1] / 'results/pdv2_geometry_interaction_20261023'
    data = json.loads((root / 'integrated_report.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    names = ['additive', 'interaction', 'temperature', 'interaction_off', 'geometry_off']
    labels = ['Additive', 'Edge interaction', 'Temperature', 'Interaction off*', 'Geometry off*']
    ap = [100 * data['final_validation'][name]['macro']['AP'] for name in names]
    bars = axes[0].barh(labels[::-1], ap[::-1], color=['#4b78a8', '#ce8d43', '#668b66', '#aaa', '#ccc'][::-1])
    axes[0].bar_label(bars, fmt='%.3f', padding=4, fontsize=10)
    axes[0].set_xlim(min(ap) - .05, max(ap) + .035)
    axes[0].set_xlabel('Final validation AP (%)')
    axes[0].set_title('One seed; fixed epoch 20\n*Frozen interaction model, inference only', fontsize=11)
    arms = ['previous_nodecay', 'interaction', 'temperature']
    bottom = np.zeros(3)
    for key, label, color in [('missing_union', 'Correct candidate missing', '#c66b5a'),
                              ('key_also_wrong', 'Key top-1 also incorrect', '#b7a27a'),
                              ('key_prefers_true_fusion_wrong', 'Key top-1 correct, fusion incorrect', '#698bb6')]:
        values = np.array([data['kitti_losses'][arm][key] for arm in arms])
        axes[1].bar(range(3), values, bottom=bottom, label=label, color=color)
        bottom += values
    for i, arm in enumerate(arms):
        axes[1].text(i, bottom[i] + .5, str(int(bottom[i])), ha='center')
    axes[1].set_ylim(0, 33)
    axes[1].set_xticks(range(3), [f'{label}\n(gained {data["kitti_losses"][arm]["gains"]})'
                              for label, arm in zip(['Prior no decay', 'Edge interaction', 'Temperature'], arms)])
    axes[1].set_ylabel('Previously correct KITTI queries lost')
    axes[1].set_title('Candidate availability and key top-1\nEach variant uses its matched training baseline', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=8, framealpha=1)
    for ext in ('png', 'pdf'):
        fig.savefig(root / ('geometry_interaction_summary.' + ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
