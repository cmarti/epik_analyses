#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm

from scipy.stats import pearsonr


if __name__ == '__main__':
    subset = 'qtls'
    environments = ['30C', 'li']

    df = pd.read_csv('datasets/{}_li_hq.csv'.format(subset), index_col=0)
    l = len(df.index[0])
    wts = ['A' * l, 'B' * l]


    df = df.join(pd.read_csv('datasets/{}_30C_hq.csv'.format(subset), index_col=0), rsuffix='_30C', lsuffix='_li').dropna()
    # df['As'] = [x.count('A') for x in df.index]
    # print(df.sort_values('As'))
    df['ENA1'] = [x[12] for x in df.index.values]
    df['pA'] = [np.sum(x == 'A' for x in seq) for seq in df.index]

    print(df['pA'].min(), df['pA'].max())
    print(df.loc[df['pA'] > 63, :])
    print(df.loc[df['pA'] < 20, :])
    exit()
    
    cols = ['y_li', 'y_30C']
    xmin, xmax = df[cols].values.min(), df[cols].values.max()
    lims = xmin, xmax
    diff = lims[1] - lims[0]
    lims = (lims[0] - 0.05 * diff, lims[1] + 0.05 * diff)
    
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))    

    df1, df2 = df.loc[df['ENA1'] == 'A', :], df.loc[df['ENA1'] == 'B', :]
    r1 = pearsonr(x=df1['y_30C'], y=df1['y_li'])[0]
    r2 = pearsonr(x=df2['y_30C'], y=df2['y_li'])[0]
    axes.scatter(x=df2['y_30C'], y=df2['y_li'], s=3, c='purple', alpha=0.1, lw=0, label='ENA1-BY (r={:.2f})'.format(r2))
    axes.scatter(x=df1['y_30C'], y=df1['y_li'], s=3, c='orange', alpha=0.1, lw=0, label='ENA1-RM (r={:.2f})'.format(r1))
    axes.legend(loc=2, fontsize=8)

    ticks = np.array([0., -0.1, -0.2, -0.3, -0.4, -0.5])
    axes.set(ylabel='Fitness in $Li^{+}$', 
             xlabel='Baseline fitness', 
             xlim=lims, ylim=lims, aspect='equal',
             xticks=ticks, yticks=ticks)
    axes.plot(lims, lims, lw=0.75, linestyle='--', c='grey')
    axes.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('plots/envs_comparison.png', dpi=300)

    print(pearsonr(df['y_30C'], df['y_li']))
    
    seqs1 = df.index[df['y_li']> -0.2].values
    seqs2 = df.index[df['y_li']< -0.2].values

    m0 = lm.alignment_to_matrix(df.index.values, to_type='probability', pseudocount=0)
    m1 = lm.alignment_to_matrix(seqs1, to_type='probability', pseudocount=0)
    m2 = lm.alignment_to_matrix(seqs2, to_type='probability', pseudocount=0)
    m = np.log2(m2 / m1)
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 1.5))
    
    lm.Logo(m, ax=axes)
    fig.savefig('plots/envs_logo.png', dpi=300)

    sel_loci = pd.read_csv('raw/qtls.tsv', sep='\t')
    sel_loci = sel_loci.loc[sel_loci['Environment'] == 'li', :]
    annotations = pd.read_csv('raw/SNP_list.csv', index_col=0)
    sel_loci = sel_loci.join(annotations, on='SNP_index')
    sel_loci['log_enrichment'] = m.iloc[:, 0].values
    sel_loci['af'] = m0.iloc[:, 0].values
    print(sel_loci.head(20))
    print(np.where(sel_loci['gene'] == 'ENA1'))
    
