#!/usr/bin/env python
from os.path import exists

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.figures.plot_utils import FIG_WIDTH


def highlight_seq_heatmap(axes, matrix, seq='DEEEIRTTNPVATEQYGSVSTNLQRGNR'):
    axes.set_clip_on(False)
    for x, c in enumerate(seq):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


if __name__ == '__main__':
    fraction_width = 0.35
    alleles_order = ['R', 'K', 'E', 'D', 'Q', 'N', 'H', 'S', 'T', 'A',
                     'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C']
    
    bcs = ['DEEEIRTTNPVATEQYGSVSTNLQRGNR', 
           'DEEEIRTTNPVATEQYGSVSTNLQRGER', 'DEEEIRTTNPVATEQYGSVSTNLQEGER']
    labels = ['WT', 'N587E', 'R585E+N587E']
    
    print('Plotting heatmap of predicted mutational effects')
    for bc, label in zip(bcs, labels):
        print('\tIn {} background'.format(label))
        fpath = 'output_new/aav.Jenga.{}_expansion.csv'.format(bc)
        data = pd.read_csv(fpath, index_col=0)
        data = data.loc[['_' not in x for x in data.index], :] # remove pairwise interaction if needed
        data['pos'] = [int(x[1:-1]) for x in data.index]
        data['allele'] = [x[-1:] for x in data.index]
        df = pd.pivot_table(data, index='allele', columns='pos', values='coef').fillna(0).loc[alleles_order, :]
        df.columns = np.arange(561, 561+28)
        
        fig, axes = plt.subplots(1, 1, figsize=(0.55 * FIG_WIDTH, 0.5 * FIG_WIDTH))
        cbar_axes = fig.add_axes([0.875, 0.3, 0.02, 0.4])
        fig.subplots_adjust(right=0.85, left=0.1, bottom=0.15)
        
        sns.heatmap(df, ax=axes, cmap='coolwarm', 
                    center=0, cbar_ax=cbar_axes,
                    vmin=-8, vmax=3,
                    cbar_kws={'label': 'Mutational effect'})
        axes.set(xlabel='Position', ylabel='Allele', aspect='equal', 
                 xticks=0.5 + np.arange(28))
        axes.set_yticklabels(axes.get_yticklabels(), rotation=0, fontsize=8)
        axes.set_xticklabels(df.columns, rotation=90, fontsize=8)
        highlight_seq_heatmap(axes, df.T, seq=bc)
        
        sns.despine(right=False, top=False)
        fig.savefig('figures/aav.mut_effs_{}.svg'.format(label))
        fig.savefig('figures/aav.mut_effs_{}.png'.format(label), dpi=300)
