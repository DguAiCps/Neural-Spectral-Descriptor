#!/usr/bin/env python3
"""attention 성분의 크기와 최종 판정 결과를 내보낼 그림으로 저장한다."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results/pdv2_attention_review_20261023'
s=json.loads((ROOT/'attention_statistics.json').read_text())['validation_sequence_mean']
r=json.loads((ROOT/'summary.json').read_text())['results']
fig,axes=plt.subplots(1,3,figsize=(12.8,3.8),layout='constrained')
x=np.arange(4);labels=['Diff L1','Diff L2','Standard L1','Standard L2']
rows=[s[a][l] for a in ('diff_linear','standard_linear') for l in ('0','1')]
for name,offset,color,label in [('content_centered_rms',-.17,'#2166ac','Feature score'),('geometry_centered_rms',.17,'#d6604d','Geometry bias')]:
 axes[0].bar(x+offset,[v[name] for v in rows],width=.32,color=color,label=label)
axes[0].set_yscale('log');axes[0].set_ylabel('Centered score RMS (log scale)')
axes[0].set_xticks(x,labels,rotation=20);axes[0].legend(fontsize=8);axes[0].set_title('Matched controls: tanh K, linear V')
axes[1].bar(x,[v['effective_neighbors_mean'] for v in rows],color=['#2166ac']*2+['#d6604d']*2)
axes[1].axhline(10,color='gray',linestyle='--',linewidth=1)
axes[1].set_ylim(0,11);axes[1].set_xticks(x,labels,rotation=20);axes[1].set_ylabel('Effective neighbors out of 10')
axes[1].set_title('Attention in the same controls')
for i,v in enumerate(rows):axes[1].text(i,v['effective_neighbors_mean']+.12,f"{v['effective_neighbors_mean']:.2f}",ha='center',fontsize=9)
values=[100*r[a]['arc_unique_b100']['macro']['AP'] for a in ('full','content','geometry','uniform')]
axes[2].plot(x,values,'o',color='#2166ac');axes[2].set_xticks(x,['Full','Feature only','Geometry only','Uniform'],rotation=20)
axes[2].set_ylim(min(values)-.08,max(values)+.08);axes[2].set_ylabel('Final validation AP (%)')
axes[2].set_title('Frozen Diff (tanh V), no retraining');axes[2].set_xlim(-.45,3.45)
for i,v in enumerate(values):axes[2].annotate(f'{v:.4f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Attention review | seed 1 | validation only')
fig.savefig(ROOT/'attention_review.png',dpi=180)
fig.savefig(ROOT/'attention_review.pdf')
plt.close(fig)
