#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as plot
import logomaker as lm

from gpmap.src.genotypes import select_genotypes
from gpmap.src.plot.mpl import get_hist_inset_axes
from gpmap.src.utils import read_edges
from gpmap.src.datasets import DataSet


if __name__ == '__main__':
    # Load visualization
    nodes = pd.read_csv('output/aav.pred_comb.nodes.csv', index_col=0)
    print(nodes)
    # edges = read_edges('output/aav.pred_comb.edges.npz')
    
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))
    
    plot.plot_visualization(axes, nodes, y='2',
                            # edges_df=edges, 
                            edges_alpha=0.02,
                            nodes_cbar=True,
                            nodes_cmap_label='log(Enrichment)',
                            nodes_cmap='viridis',
                            sort_ascending=True, fontsize=11)
    lims = (-3, 1.5)
    axes.set(aspect='equal', xlim=lims, ylim=lims)
    axes.set_xticks(axes.get_yticks())
    axes.grid(alpha=0.1)
    fig.tight_layout()
    fig.savefig('plots/aav.pred_comb.visualization.png', dpi=300)
    
    fig, axes = plt.subplots(1, 1, figsize=(3, 1.75))
    sns.histplot(nodes['function'], bins=30, color='grey', linewidths=1, edgecolor='black', ax=axes,
                 stat='probability')
    axes.set(xlabel='Genotype log(Enrichment)')
    fig.tight_layout()
    fig.savefig('plots/aav.pred_comb.function_distribution.png', dpi=300)
    
    
    seqs = [nodes.index[(nodes['1'] < -1.5) & (nodes['function'] >= 0)].values,
            nodes.index[(nodes['1'] >  0.6) & (nodes['function'] >= 0)].values,
            nodes.index[(nodes['2'] >  0.6) & (nodes['function'] >= 0)].values]
    ms = [lm.alignment_to_matrix(s, to_type='probability', pseudocount=0) for s in seqs]
    nplots = len(ms)
    fig, subplots = plt.subplots(nplots, 1, figsize=(3, 1.25 * nplots))
    
    for m, axes in zip(ms, subplots):
        print(m)
        lm.Logo(m, ax=axes, color_scheme='chemistry')
        axes.set(xticks=np.arange(4), xticklabels=['582', '585', '587', '588'],
                 xlabel='Position', ylabel='Probability')
    
    fig.tight_layout()
    fig.savefig('plots/aav.pred_comb.peaks.png', dpi=300)