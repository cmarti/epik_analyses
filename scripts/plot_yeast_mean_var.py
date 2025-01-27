#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

    

if __name__ == '__main__':
    data = pd.read_csv('datasets/qtls_li_hq.csv', index_col=0)
    data['y_sd'] = np.sqrt(data['y_var'])
    print(data)
    
    fig, axes= plt.subplots(1, 1, figsize=(4, 3.5))
    axes.scatter(data['y'], data['y_sd'], c='black', lw=0, alpha=0.05, s=5)
    
    axes.set(xlabel='Mean', ylabel='SD', yscale='log',
             ylim=(1e-2, 1))
    axes.grid(alpha=0.2)
    
    fig.tight_layout()
    fig.savefig('plots/mean_std.png', dpi=300)