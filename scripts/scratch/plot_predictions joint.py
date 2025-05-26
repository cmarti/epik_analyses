#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from tqdm import tqdm
from scipy.stats import pearsonr


def plot_scatter(x, y, axes, vmin=0, vmax=3):
    r2 = pearsonr(x, y)[0] ** 2
    rmse = np.sqrt(np.mean((x - y) ** 2))
    
    lims = min(x.min(), y.min()), max(x.max(), y.max())
    diff = lims[1] - lims[0]
    lims = (lims[0] - 0.05 * diff, lims[1] + 0.05 * diff)
    
    # bins = np.linspace(lims[0], lims[1], 100)
    # H, xbins, ybins = np.histogram2d(x=x, y=y, bins=bins)
    # im = axes.imshow(np.log10(H.T[::-1, :]), cmap='viridis',
    #                  extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
    #                  vmin=vmin, vmax=vmax)
    axes.scatter(x, y, c='black', s=3, alpha=0.2, lw=0)
    axes.plot(lims, lims, lw=0.75, linestyle='--', c='grey')
    axes.text(0.95, 0.05, '$R^2$={:.2f}\nRMSE={:.3f}'.format(r2, rmse),
              transform=axes.transAxes, ha='right', va='bottom')
    ticks = np.array([0., -0.1, -0.2, -0.3, -0.4, -0.5])
    axes.set(xlabel=r'Predicted fitness',
             ylabel=r'Measured fitness',
             xlim=lims, ylim=lims, aspect='equal',
             xticks=ticks, yticks=ticks)
    
    axes.grid(alpha=0.2)
    return()
    

if __name__ == '__main__':
    ds = 'qtls_merged_hq'
    kernel = 'Rho'

    results = []    
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))    
    
    pred = []
    data = pd.read_csv('datasets/{}.csv'.format(ds), index_col=0)
    for i in [51, 52, 53]:
        fpath = 'output/{}.{}.{}.test_pred.csv'.format(ds, i, kernel)
        if not exists(fpath):
            continue
        pred.append(pd.read_csv(fpath, index_col=0).join(data).dropna())
    pred = pd.concat(pred, axis=0)
    x, y = pred['y_pred'].values, pred['y'].values
    plot_scatter(x, y, axes, vmax=3)
        
    fig.tight_layout()
    fig.savefig('plots/{}.scatter.png'.format(ds), dpi=300)
        
