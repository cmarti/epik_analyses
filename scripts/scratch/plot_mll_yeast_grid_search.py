#!/usr/bin/env python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    data = pd.read_csv('output/exponential_yeast_grid_search.csv')
    print(data.sort_values('mll', ascending=False).iloc[:20, :])
    data = pd.pivot_table(data, index='theta', columns='log_var', values='mll')
    X, Y = np.meshgrid(data.columns.values, data.index.values)
    
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    
    axes.contourf(X, Y, data, cmap='viridis', levels=200)
    # sns.heatmap(data.iloc[::-1, :], ax=axes, cmap='viridis')
    axes.set(ylabel=r'$\log \rho$',
             xlabel=r'$\log \sigma^2$', #xticks=[], yticks=[], 
            #  aspect='equal',
             )
    fig.tight_layout()
    fig.savefig('plots/yeast_exponential_grid_search.png', dpi=300)
    
