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
    dataset = 'smn1'
    kernel_name = 'ARD'
    
    # Load kernel function
    fpath = 'output/{}.{}_kernel_function.csv'.format(dataset, kernel_name)
    kernel = pd.read_csv(fpath, index_col=0)
    print(kernel.loc[kernel.columns, :])
    
    # Load visualization
    nodes = pd.read_csv('output/{}.{}.nodes.csv'.format(dataset, kernel_name), index_col=0)
    edges = read_edges('output/{}.{}.edges.npz'.format(dataset, kernel_name))
    nodes = nodes.join(kernel)
    
    fig, subplots = plt.subplots(1, 4, figsize=(12, 3.5),
                                 sharex=True, sharey=True)
    
    axes = subplots[0]
    plot.plot_visualization(axes, nodes, edges_df=edges,  
                            edges_alpha=0.05, nodes_vmax=100, nodes_vmin=0,
                            nodes_cbar=False, sort_ascending=True, fontsize=11)
    axes.set(title='Percent Spliced In (%)')
    
    
    ax = get_hist_inset_axes(axes, pos=(0.45, 0.30), height=0.2, width=0.5)
    seqs = nodes.loc[nodes['1'] > 5, :].index.values
    m = lm.alignment_to_matrix(seqs, to_type='probability', pseudocount=0)
    lm.Logo(m, ax=ax)
    ax.set(xlabel='', ylabel='', xticks=[], yticks=[])
    
    ax = get_hist_inset_axes(axes, pos=(0.15, 0.05), height=0.2, width=0.5)
    seqs = nodes.loc[nodes['2'] < -2, :].index.values
    m = lm.alignment_to_matrix(seqs, to_type='probability', pseudocount=0)
    lm.Logo(m, ax=ax)
    ax.set(xlabel='', ylabel='', xticks=[], yticks=[])
    
    
    for axes, seq in zip(subplots[1:], ['UCUUAAGU', 'CAGUUCAA', 'GGUCGUUU']):
        plot.plot_visualization(axes, nodes,
                                edges_df=edges, 
                                edges_alpha=0.05,
                                nodes_vmax=1, nodes_vmin=-0.1,
                                nodes_cbar=False,
                                nodes_color=seq,
                                nodes_cmap='binary',
                                sort_by=seq, sort_ascending=True, fontsize=11)
        axes.set(title='K({}, x)'.format(seq))
        axes.set_ylabel('')
        
    fig.tight_layout()
    fig.savefig('plots/{}.visualization.{}.png'.format(dataset, kernel_name), dpi=300)