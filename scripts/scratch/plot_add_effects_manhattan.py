#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from itertools import combinations
from scipy.stats import mannwhitneyu
from torch.distributions.transforms import CorrCholeskyTransform


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    id='53'
    environment = 'li'

    sel_loci = pd.read_csv('raw/qtls.tsv', sep='\t')
    sel_loci = sel_loci.loc[sel_loci['Environment'] == environment, :]
    annotations = pd.read_csv('raw/SNP_list.csv', index_col=0)
    sel_loci = sel_loci.join(annotations, on='SNP_index')
    sel_loci['chr'] = ['chr{}'.format(i) for i in sel_loci['Chromosome']]
    chr_sizes = pd.read_csv('raw/chr_sizes.csv', index_col=0)['size']
    chr_x = np.cumsum(chr_sizes)
    sel_loci['x'] = chr_x.loc[sel_loci['chr']].values + sel_loci['Position (bp)']
    sel_loci['abs_eff'] = np.abs(sel_loci['Effect'])

    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    axes.scatter(sel_loci['x'], sel_loci['Effect'], c='black', s=7.5, lw=0)

    # Color by chromosome
    ylim = (-0.2, 0.2)
    axes.grid(alpha=0.1)
    for i in range(0, chr_x.shape[0], 2):
        left, right = chr_x.iloc[i], chr_x.iloc[i+1]
        axes.fill_between(x=(left, right), y1=ylim[0], y2=ylim[1], color='grey', alpha=0.2, lw=0)

    # Get chromosome label positions
    chr_pos = []
    prev_pos = 0
    for pos in chr_x:
        chr_pos.append((pos + prev_pos) / 2)
        prev_pos = pos
    chr_labels = ['chr{}'.format(i+1) for i in range(len(chr_pos))]

    axes.hlines(0, 0, chr_x.max(), lw=0.5, colors='grey', linestyles='--')
    axes.set(title='Growth rate in {}'.format(environment.capitalize()),
             xlabel='Chromomsome position (bp)', ylabel='Mutational effect',
             ylim=ylim, xlim=(0, chr_x.max()),
             xticks=chr_pos)
    axes.set_xticklabels(chr_labels, rotation=45)

    labels = sel_loci.dropna(subset=['gene'])
    shifts = [1, -1, 1]

    n = 10
    labels = sel_loci.dropna(subset=['gene']).sort_values('abs_eff', ascending=False).iloc[:n, :]
    dxs = [-1, 1, -1, 2, -1, -1, -0.5, 0.5, 1.5, -3]
    dys = [1, 1, 2, 2, 1, 2, 2, 1.5, 1, 1]
    for x, y, label, dx, dy in zip(labels['x'], labels['Effect'], labels['gene'], dxs, dys):
        dy = np.abs(dy) if y > 0 else - np.abs(dy)
        xtext = x  + 1e5 * dx
        ytext = y + 0.035 * dy
        ha = 'left' if dx > 0 else 'right'
        va = 'bottom' if dy > 0 else 'top'
        axes.annotate(label, xy=(x, y), ha=ha,
                      xytext=(xtext, ytext), fontsize=7,
                      arrowprops=dict(facecolor='black', shrink=1,
                                      width=0.25, headwidth=2, headlength=2))

    fig.tight_layout()
    fig.savefig('plots/{}.additive_effects.png'.format(environment), dpi=300)
    fig.savefig('plots/{}.additive_effects.pdf'.format(environment), dpi=300)
