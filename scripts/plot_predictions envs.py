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
    bins = np.linspace(lims[0], lims[1], 100)
    diff = lims[1] - lims[0]
    lims = (lims[0] - 0.05 * diff, lims[1] + 0.05 * diff)
    
    H, xbins, ybins = np.histogram2d(x=x, y=y, bins=bins)
    im = axes.imshow(np.log10(H.T[::-1, :]), cmap='viridis',
                     extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
                     vmin=vmin, vmax=vmax)
    axes.plot(lims, lims, lw=0.5, linestyle='--', c='black')
    axes.text(0.95, 0.05, '$R^2$={:.2f}\nRMSE={:.3f}'.format(r2, rmse),
              transform=axes.transAxes, ha='right', va='bottom')
    axes.set(xlabel=r'Predicted phenotype $y_{pred}$',
             ylabel=r'Observed phenotype $y_{obs}$',
             xlim=lims, ylim=lims)
    axes.grid(alpha=0.2)
    return(im)
    

if __name__ == '__main__':
    datasets = [('qtls_30C_hq', 'qtls_li_hq'), ('qtls_merged_hq', )]
    kernel = 'Rho'
    labels = ['1-D Fitness', '2D-Fitness']

    results = []    
    fig, subplots = plt.subplots(1, 2, figsize=(8, 3.), sharex=True, sharey=True)
    
    for axes, dataset, label in zip(subplots, datasets, labels):
        print('Evaluating dataset: {}'.format(dataset))
        # Load full dataset

        pred = []
        for ds in dataset:
            data = pd.read_csv('datasets/{}.csv'.format(ds), index_col=0)
            for i in [18, 19, 20]:
                fpath = 'output/{}.{}.{}.test_pred.csv'.format(ds, i, kernel)
                if not exists(fpath):
                    continue
                pred.append(pd.read_csv(fpath, index_col=0).join(data).dropna())
        pred = pd.concat(pred, axis=0)
        x, y = pred['y_pred'].values, pred['y'].values
        im = plot_scatter(x, y, axes, vmax=3)
        axes.set_title(label)
        fig.colorbar(im, label='Frequency', shrink=0.8)
        axes = fig.axes[-1]
        axes.set_yticklabels(['{:.0f}'.format(10**i) for i in axes.get_yticks()])
        
    fig.tight_layout()
    fig.savefig('plots/envs.scatter.png', dpi=300)
        
