#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


def read_decay_factors(dataset, kernel='Rho'):
    fpath = 'output_gpu/yeast.{}.{}.model_params.pth'.format(dataset, kernel)
    params = torch.load(fpath, map_location=torch.device('cpu'))
    logit_rho = params['covar_module.logit_rho'].numpy()
    log_p = params['covar_module.log_p'].numpy()
    
    rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
    p = np.exp(log_p)
    p = p / np.expand_dims(p.sum(1), 1)
    eta = (1 - p) / p
    
    decay_factor = 1 - (1 - rho) / (1 + eta * rho)
    return(decay_factor)


def manhattan_heatmap(values, annotations):
    fig, axes = plt.subplots(1, 1, figsize=(9, 4.))
    cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(values, ax=axes, cmap='Blues', 
                # vmin=0,
                vmax=2,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'$\log_{10}$(Decay factor)'})
    
    chroms = annotations.groupby(['chr'])['pos'].mean()
    chroms_bounds = annotations.groupby(['chr'])['pos'].max()
    ylims = axes.get_ylim()
    axes.vlines(chroms_bounds.values+1, 
                ymin=ylims[0], ymax=ylims[1], lw=1, colors='black')
    
    axes.set(title='Yeast growth environments',
             xlabel='Position', ylabel='Environment',
             xticks=chroms.values)
    axes.set_xticklabels(chroms.index, rotation=45)
    sns.despine(ax=axes, right=False, top=False)
    
    labels = annotations.dropna(subset=['gene'])
    axes = axes.twiny()
    axes.set_xticks(labels['pos'])
    axes.set_xticklabels(labels['gene'], rotation=90, fontsize=6)

    fig.savefig('plots/yeast_envs.decay_factors.png', dpi=300)
    fig.savefig('plots/yeast_envs.decay_factors.pdf', dpi=300)    
    

if __name__ == '__main__':
    annotations = pd.read_csv('loci_annotations.csv', index_col=0)
    annotations['pos'] = np.arange(annotations.shape[0])
    
    environments = [line.strip() for line in open('environments.txt')]
    kernel = 'Rho'
    
    decay_factors = []
    labels = []
    for environment in environments:
        try:
            f = read_decay_factors(environment, kernel=kernel)
            labels.append(environment)
            decay_factors.append(np.log10(f * 100).mean(1))
        except:
            continue
    decay_factors = pd.DataFrame(decay_factors, index=labels)
    manhattan_heatmap(decay_factors, annotations)
    
    annotations = annotations.join(decay_factors.T)
    annotations.to_csv('loci_decay_rates.csv')
    
    
