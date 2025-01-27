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
    axes.text(0.95, 0.05, '$R^2$={:.2f}\nRMSE={:.2f}'.format(r2, rmse),
              transform=axes.transAxes, ha='right', va='bottom')
    axes.set(xlabel=r'Predicted phenotype $y_{pred}$',
             ylabel=r'Observed phenotype $y_{obs}$',
             xlim=lims, ylim=lims)
    axes.grid(alpha=0.2)
    return(im)
    

if __name__ == '__main__':

    datasets = ['yeast_30C', 'yeast_li', 'smn1', 'gb1']
    datasets = ['smn1']
    kernels = ['VC', 'Jenga']
    labels = ['Variance Component', 'Jenga model']

    results = []    
    for dataset in datasets:
        print('Evaluating dataset: {}'.format(dataset))
        # Load full dataset
        data = pd.read_csv('datasets/{}.csv'.format(dataset), index_col=0)


        fig, subplots = plt.subplots(1, 3, figsize=(12, 3.), sharex=True, sharey=True)

        for axes, kernel, label in zip(subplots, kernels, labels):
            pred = []
            for i in [51, 52, 53]:
                fpath = 'output/{}.{}.{}.test_pred.csv'.format(dataset, i, kernel)
                if not exists(fpath):
                    print(fpath)
                    continue
                pred.append(pd.read_csv(fpath, index_col=0).join(data).dropna())
                print(pred)
            pred = pd.concat(pred, axis=0)
            x, y = pred['coef'].values, pred['y'].values
            im = plot_scatter(x, y, axes)
            axes.set_title(label)
            fig.colorbar(im, label='Frequency', shrink=0.8)
            axes = fig.axes[-1]
            axes.set_yticklabels(['{:.0f}'.format(10**i) for i in axes.get_yticks()])
        
        fig.tight_layout()
        fig.savefig('plots/{}.scatter.png'.format(dataset), dpi=300)
        
