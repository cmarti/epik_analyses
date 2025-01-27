#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gpmap.src.plot.mpl as plot
from itertools import combinations
from gpmap.src.space import SequenceSpace

if __name__ == '__main__':
    covs = pd.read_csv('smn1.vj_covariances.csv'.format(), index_col=0).sort_index()
    covs['x'] = np.random.normal(covs['d'], scale=0.075)
    covs['prior_ns'] = covs['map_ns']
    space = SequenceSpace(X=covs.index.values, y=covs['data'].values)
    edges = space.get_edges_df()
    
    fig, axes = plt.subplots(1, 1, figsize=(3.25, 3))

    lim = (-0.1, 1.05)
    plot.plot_visualization(axes, covs, edges_df=edges, x='x', y='data',
                            nodes_size=7.5, nodes_color='grey', nodes_alpha=0.6,
                            edges_color='lightgrey', edges_alpha=0.4)
    
    df = covs[['d', 'data', 'data_ns']]
    df.columns = ['d', 'c', 'n']
    df['c'] = df['c'] * df['n']
    df = df.groupby(['d'])[['c', 'n']].sum().reset_index()
    df['c'] = df['c'] / df['n']

    axes.plot(df['d'], df['c'], lw=1.5, color='black', zorder=5)
    axes.scatter(df['d'], df['c'], s=15, color='black', zorder=5)
    axes.set(xlabel='Hamming distance', ylabel='Correlation',
                xticks=np.arange(space.seq_length + 1), ylim=lim)
    axes.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig('plots/smn1_distance_correlations.png', dpi=300)
    fig.savefig('plots/smn1_distance_correlations.svg', dpi=300)


    
