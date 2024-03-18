#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


def read_decay_factors(dataset, id, kernel='Rho'):
    params = torch.load('output_gpu/{}.{}.{}.test_pred.csv.model_params.pth'.format(dataset, id, kernel),
                        map_location=torch.device('cpu'))
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
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC',
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
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC',
                      'aav': 'AAV2 Capside'}
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 2.5))
    cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(np.log10(values * 100), ax=axes, cmap='Blues', 
                # vmin=0,
                vmax=2,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'$\log_{10}$(Decay factor)'})
    
    axes.set(title=dataset_labels[dataset],
             xlabel='Position', ylabel='Training set',
             xticks=[], yticks=[])
    sns.despine(ax=axes, right=False, top=False)
    axes.grid(alpha=0.2)
    
    fig.savefig('plots/{}.decay_factors_heatmap.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.decay_factors_heatmap.pdf'.format(dataset), dpi=300)    


if __name__ == '__main__':
    kernel = 'ARD'
    
    for dataset, kernel in [('smn1', 'ARD'),
                            ('gb1', 'ARD'),
                            ('aav', 'ARD'),
                            ('yeast', 'Rho'),
                            ('yeast.37C', 'Rho')]:
        decay_factors = []
        for i in range(54):
            try:
                try:
                    f = read_decay_factors(dataset, id=str(i), kernel=kernel)
                except KeyError:
                    continue
                if kernel == 'Rho':
                    f = f.mean(1)
                elif kernel == 'ARD':
                    f = f.flatten()
                else:
                    raise ValueError('Unknown kernel: {}'.format(kernel))
                decay_factors.append(f)
            except FileNotFoundError:
                continue
        decay_factors = np.vstack(decay_factors)
        manhattan_heatmap(dataset, decay_factors)
