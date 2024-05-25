#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from os.path import exists
from epik.src.utils import seq_to_binary


def manhattan_heatmap(betas, environments):
    values = np.log(np.abs(betas[environments].T))
    fig, axes = plt.subplots(1, 1, figsize=(9, 4.))
    cbar_axes = fig.add_axes([0.88, 0.275, 0.02, 0.5])
    fig.subplots_adjust(right=0.85, left=0.12, bottom=0.2)
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(values, ax=axes, cmap='Blues', 
                vmin=-6,
                # vmax=2,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'$\log(\hat\beta)$'})
    
    chroms = betas.groupby(['chr'])['pos'].mean()
    chroms_bounds = betas.groupby(['chr'])['pos'].max()
    ylims = axes.get_ylim()
    axes.vlines(chroms_bounds.values+1, 
                ymin=ylims[0], ymax=ylims[1], lw=0.5,
                colors='black')
    
    axes.set(title='Yeast growth environments',
             xlabel='Position', ylabel='Environment',
             xticks=chroms.values)
    axes.set_xticklabels(chroms.index, rotation=45)
    sns.despine(ax=axes, right=False, top=False)
    
    labels = betas.dropna(subset=['gene'])
    axes = axes.twiny()
    axes.set_xticks(labels['pos'])
    axes.set_xticklabels(labels['gene'], rotation=90, fontsize=6)

    sns.despine(ax=cbar_axes, right=False, top=False, bottom=False, left=False)

    fig.savefig('plots/yeast_envs.beta.png', dpi=300)
    fig.savefig('plots/yeast_envs.beta.pdf', dpi=300)    
    

if __name__ == '__main__':
    betas = pd.read_csv('loci_betas_annotations.csv', index_col=0)
    environments = [line.strip() for line in open('environments.txt')]
    manhattan_heatmap(betas, environments)
    
    
    
