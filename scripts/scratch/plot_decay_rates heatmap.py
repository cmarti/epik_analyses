#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from itertools import combinations
from torch.distributions.transforms import CorrCholeskyTransform


def read_decay_factors(dataset, id, kernel='Rho'):
    fpath = 'output/{}.{}.{}.test_pred.csv.model_params.pth'.format(dataset, id, kernel)
    params = torch.load(fpath, map_location=torch.device('cpu'))

    if kernel == 'GeneralProduct':
        theta = params['covar_module.base_kernel.theta']
        to_L = CorrCholeskyTransform()
        Ls = [to_L(x) for x in theta]
        corrs = [L @ L.T for L in Ls]
        idxs = np.arange(corrs[0].shape[0])
        decay_factor = 1 - np.array([[c[i, j]
                                      for i,j in combinations(idxs, 2)]
                                      for c in corrs]).flatten()
    else:
        logit_rho = params['covar_module.logit_rho'].numpy()
        log_p = params['covar_module.log_p'].numpy()
        rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
        p = np.exp(log_p)
        p = p / np.expand_dims(p.sum(1), 1)
        eta = (1 - p) / p
        decay_factor = 1 - (1 - rho) / (1 + eta * rho)

    return(decay_factor)


def plot_hist(dataset, values):
    dataset_labels = {'gb1': 'Protein GB1', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast_li': 'Yeast growth in Li',
                      'yeast_37C': 'Yeast growth at 37ºC',
                      'aav': 'AAV2 Capside'}
    
    fig, axes = plt.subplots(1, 1, figsize=(3, 2.5))
    sns.histplot(np.log(values), ax=axes, bins=30)
    axes.set(title=dataset_labels[dataset],
             xlabel='log(Decay factor)',
             ylabel='# loci')
    sns.despine(ax=axes, right=False, top=False)
    
    fig.tight_layout()
    fig.savefig('plots/{}.decay_factors_hist.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.decay_factors_hist.pdf'.format(dataset), dpi=300)


def manhattan_heatmap(dataset, values):
    dataset_labels = {'gb1': 'Protein GB1', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast_li': 'Yeast growth in Li',
                      'qtls_li': 'Yeast growth in Li',
                      'yeast_37C': 'Yeast growth at 37ºC',
                      'yeast_30C': 'Yeast growth at 30ºC',
                      'aav': 'AAV2 Capside'}
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 2.5))
    cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(values * 100, ax=axes, cmap='Blues', 
                vmin=0,
                vmax=100,
                cbar_ax=cbar_axes,
                cbar_kws={'label': 'Decay rate'})
    
    axes.set(title=dataset_labels[dataset],
             xlabel='Position', ylabel='Training set',
             xticks=[], yticks=[])
    sns.despine(ax=axes, right=False, top=False)
    axes.grid(alpha=0.2)
    
    return(fig)


if __name__ == '__main__':
    for dataset, kernel in [
                            # ('smn1', 'GeneralProduct'),
                            # ('gb1', 'ARD'),
                            # ('aav', 'ARD'),
                            # ('yeast_30C', 'Rho'),
                            ('qtls_li', 'Rho'),
                            # ('yeast_li', 'Rho'),
                            # ('yeast_37C', 'Rho'),
                            ]:
        decay_factors = []
        for i in range(89):
            try:
                try:
                    f = read_decay_factors(dataset, id=str(i), kernel=kernel)
                except KeyError:
                    continue
                if kernel == 'Rho':
                    f = f.mean(1)
                elif kernel == 'ARD':
                    f = f.flatten()
                elif kernel == 'GeneralProduct':
                    f = f
                else:
                    raise ValueError('Unknown kernel: {}'.format(kernel))
                print(f.max())
                decay_factors.append(f)
            except FileNotFoundError:
                continue
        decay_factors = np.vstack(decay_factors)
        fig = manhattan_heatmap(dataset, decay_factors)

        fig.savefig('plots/{}.{}.decay_factors_heatmap.png'.format(dataset, kernel), dpi=300)
        fig.savefig('plots/{}.{}.decay_factors_heatmap.pdf'.format(dataset, kernel), dpi=300)    
