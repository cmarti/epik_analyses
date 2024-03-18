#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gpmap.src.plot.mpl as plot
from itertools import combinations
from gpmap.src.space import SequenceSpace

if __name__ == '__main__':
    for dataset in ['smn1', 'gb1']:

        covs = pd.read_csv('{}.vj_covariances.csv'.format(dataset), index_col=0).sort_index()
        covs['x'] = np.random.normal(covs['d'], scale=0.075)
        covs['prior_ns'] = covs['map_ns']
        space = SequenceSpace(X=covs.index.values, y=covs['data'].values)
        edges = space.get_edges_df()
        
        fig, subplots = plt.subplots(2, 3, figsize=(9, 6))

        cols = ['data', 'prior', 'map']
        lim = (-0.05, 1.05)
        for i, field in enumerate(cols):

            axes = subplots[0, i]
            plot.plot_visualization(axes, covs, edges_df=edges, x='x', y=field,
                                    nodes_size=7.5, nodes_color='grey', nodes_alpha=0.3,
                                    edges_color='lightgrey', edges_alpha=0.2)
            
            df = covs[['d', field, '{}_ns'.format(field)]]
            df.columns = ['d', 'c', 'n']
            df['c'] = df['c'] * df['n']
            df = df.groupby(['d'])[['c', 'n']].sum().reset_index()
            df['c'] = df['c'] / df['n']

            axes.plot(df['d'], df['c'], lw=1.5, color='black', zorder=5)
            axes.scatter(df['d'], df['c'], s=15, color='black', zorder=5)

            axes.set(xlabel='Hamming distance', ylabel='Correlation', title=field.upper(),
                     xticks=np.arange(space.seq_length + 1), ylim=lim)
            axes.grid(alpha=0.2)

        for i, (field1, field2) in enumerate(combinations(['data', 'prior', 'map'], 2)):

            axes = subplots[1, i]
            axes.scatter(covs[field1], covs[field2], s=10, c='black', alpha=0.2, lw=0)
            axes.plot(lim, lim, linestyle=':', c='grey', alpha=0.3)
            axes.set(xlabel='{} correlation'.format(field1.upper()),
                     ylabel='{} correlation'.format(field2.upper()),
                     ylim=lim, xlim=lim)
            axes.grid(alpha=0.2)
        
        fig.tight_layout()
        fig.savefig('plots/{}.vj_correlations.png'.format(dataset), dpi=300)


        
