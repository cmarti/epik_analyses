#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from itertools import combinations
    

if __name__ == '__main__':
    loci_data = pd.read_csv('results/qtls_li_hq_annotations.csv')
    backgrounds = ['AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
                   'BBBBBBBBBBBBABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB',
                   'AAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
                   'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB']
    bcs = ['RM', 'BY', 'RM', 'BY']
    ena1 = ['RM', 'RM', 'BY', 'BY']
    
    data = []
    for seq, bc, ena in zip(backgrounds, bcs, ena1):
        fpath = 'output/qtls_li_hq.Connectedness.{}_expansion.csv'.format(seq)
        df = pd.read_csv(fpath, index_col=0)
        df = df.loc[['_' not in x for x in df.index], :]
        idx = np.array([x.startswith('B') for x in df.index])
        df.index = [x[-1] + x[1:-1] + x[0] if x.startswith('B') else x
                    for x in df.index]
        df.loc[idx, 'coef'] = -df.loc[idx, 'coef']
        df.loc[idx, 'lower_ci'] = -df.loc[idx, 'lower_ci']
        df.loc[idx, 'upper_ci'] = -df.loc[idx, 'upper_ci']
        df.columns = ['{}_{}_ena1{}'.format(c, bc, ena) for c in df.columns]
        data.append(df)
    data = pd.concat(data, axis=1)
    cols = ['coef_RM_ena1RM', 'coef_BY_ena1RM', 'coef_RM_ena1BY', 'coef_BY_ena1BY']
    data['idx'] = [int(x[1:-1]) for x in data.index]
    data = data.join(loci_data, on='idx', rsuffix='_locus').set_index('gene')
    print(data.loc['ENA1', cols])
    print(data.loc[np.isin(data['idx_locus'], [8, 33, 56, 59, 69, 74]), cols])
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 1.75))
    
    sns.heatmap(data[cols].T, ax=axes, cmap='seismic', center=0,
                cbar_kws={'label': 'BY - RM effect'})
    axes.set(xlabel='Locus', yticklabels=['ENA1-RM in RM background',
                                          'ENA1-RM in BY background',
                                          'ENA1-BY in RM background',
                                          'ENA1-BY in BY background'])
    axes.set(xticks=np.arange(83) + 0.5)
    axes.set_xticklabels(data.index.fillna(''), rotation=90, fontsize=9)
    
    fig.tight_layout()
    fig.savefig('plots/mut_effs_backgrounds.png', dpi=300)
    
    # Scatterplots
    fig, subplots = plt.subplots(2, 3, figsize=(3 * 3, 3 * 2), sharex=True, sharey=True)
    subplots = subplots.flatten()
    labels = ['RM_ena1RM', 'RM_ena1BY', 'BY_ena1RM', 'BY_ena1BY']
    bcs_list = list(combinations(labels, 2))
    
    for axes, (bc1, bc2) in zip(subplots, bcs_list):
        x, y = data['coef_{}'.format(bc1)], data['coef_{}'.format(bc2)]
        axes.scatter(x, y, s=5, lw=0, c='black')
        # axes.errorbar(x, y,
        #             xerr=2 * df['stderr'],
        #             yerr=2 * df['stderr_2'],
        #             fmt='', color='black', alpha=0.25, markersize=3, elinewidth=0.5, lw=0)
        axes.axline((0, 0), (0.01, 0.01), linestyle='--', lw=0.75, color='grey')
        axes.axvline(0, linestyle='--', lw=0.75, color='grey')
        axes.axhline(0, linestyle='--', lw=0.75, color='grey')
        
        axes.set(xlabel='Mutational effect in {}'.format(bc1),
                ylabel='Mutational effect in {}'.format(bc2),
                aspect='equal')
        axes.grid(alpha=0.2)
        
    fig.tight_layout()
    fig.savefig('plots/yeast_mut_effs_bcs.png', dpi=300)
    
    
    
            
    
    