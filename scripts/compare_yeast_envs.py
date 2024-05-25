#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm


if __name__ == '__main__':
    subset = 'qtls'
    environments = ['30C', 'li']

    df = pd.read_csv('datasets/{}_li_hq.csv'.format(subset), index_col=0)
    df = df.join(pd.read_csv('datasets/{}_30C_hq.csv'.format(subset), index_col=0), rsuffix='_30C', lsuffix='_li').dropna()
    df['As'] = [x.count('A') for x in df.index]
    l = len(df.index[0])
    wts = ['A' * l, 'B' * l]
    print(df.sort_values('As'))
    
    cols = ['y_li', 'y_30C']
    xmin, xmax = df[cols].values.min(), df[cols].values.max()
    lims = xmin, xmax
    bins = np.linspace(xmin, xmax, 100)
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))    
    axes.scatter(x=df['y_30C'], y=df['y_li'], s=3, c='black', alpha=0.1, lw=0)
    axes.set(ylabel='Fitness in $Li^{+}$', 
             xlabel='Baseline fitness', 
             xlim=lims, ylim=lims, aspect='equal')
    axes.plot(lims, lims, lw=0.75, linestyle='--', c='grey')
    axes.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('plots/envs_comparison.png', dpi=300)
    
    seqs = df.index[df['y_li']< -0.2].values
    m = lm.alignment_to_matrix(seqs, to_type='probability', pseudocount=0)
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 1.5))
    
    lm.Logo(m, ax=axes)
    fig.savefig('plots/envs_logo.png', dpi=300)
    
