#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def highlight_seq_heatmap(axes, matrix, seq='DEEEIRTTNPVATEQYGSVSTNLQRGNR'):
    axes.set_clip_on(False)
    for x, c in enumerate(seq):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


if __name__ == '__main__':
    bc = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    alleles_order = ['R', 'K', 'E', 'D', 'Q', 'N', 'H', 'S', 'T', 'A',
                     'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C']
    data = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bc), index_col=0)
    data = data.loc[['_' not in x for x in data.index], :]
    data['pos'] = [int(x[1:-1]) for x in data.index]
    data['allele'] = [x[-1:] for x in data.index]
    df = pd.pivot_table(data, index='allele', columns='pos', values='coef').fillna(0).loc[alleles_order, :]
    df.columns = np.arange(561, 561+28)
    
    fig, axes = plt.subplots(1, 1, figsize=(5.5, 4.))
    cbar_axes = fig.add_axes([0.875, 0.3, 0.02, 0.4])
    fig.subplots_adjust(right=0.85, left=0.1)
    
    sns.heatmap(df, ax=axes, cmap='coolwarm', 
                center=0, cbar_ax=cbar_axes,
                vmin=-8, vmax=3,
                cbar_kws={'label': 'Mutational effect'})
    axes.set(xlabel='Position', ylabel='Allele')
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(axes, df.T, seq=bc)
    
    sns.despine(right=False, top=False)
    fig.savefig('plots/aav.mut_effs_{}.png'.format(bc), dpi=300)
