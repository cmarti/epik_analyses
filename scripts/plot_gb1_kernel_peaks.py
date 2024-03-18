#!/usr/bin/env python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from gpmap.src.datasets import DataSet
import gpmap.src.plot.mpl as plot


if __name__ == '__main__':
    id = '29'
    dataset = 'gb1'
    kernel_name = 'ARD'
    
    fpath = 'output_gpu/{}.{}.{}_kernel_function.csv'.format(dataset, id, kernel_name)
    kernel = pd.read_csv(fpath, index_col=0)
    print(kernel.loc[kernel.columns, :])
    
    gb1 = DataSet('gb1')
    nodes = gb1.nodes.join(kernel)
    
    extent = (nodes['1'].min(), nodes['1'].max(),
              nodes['2'].min(), nodes['2'].max())
    fig, subplots = plt.subplots(1, 3, figsize=(15, 4))
    
    for axes, seq in zip(subplots, kernel.columns):
        plot.plot_visualization(axes, nodes,
                                # edges_df=gb1.edges, 
                                edges_alpha=0.005,
                                nodes_vmax=1, nodes_vmin=0,
                                nodes_color=seq, nodes_cmap_label='K({}, x)'.format(seq),
                                sort_by=seq, sort_ascending=True, fontsize=11)
        
    fig.tight_layout()
    fig.savefig('plots/gb1.visualization.{}_kernel.png'.format(kernel_name), dpi=300)