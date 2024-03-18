#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


def read_params(dataset, id):
    fpath = 'output_gpu/{}.{}.mavenn_linear.params.csv'.format(dataset, id)
    params = pd.read_csv(fpath, index_col=0).max(1).values
    return(params)


def manhattan_heatmap(values):
    fig, axes = plt.subplots(1, 1, figsize=(8, 4.))
    cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(values, ax=axes, cmap='Blues', 
                # vmin=0,
                # vmax=2,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'Mutational effect'})
    
    axes.set(title='Yeast growth environments',
             xlabel='Position', ylabel='Environment',
             xticks=[])
    sns.despine(ax=axes, right=False, top=False)
    axes.grid(alpha=0.2)
    
    fig.savefig('plots/yeast_envs.mut_eff.png', dpi=300)
    fig.savefig('plots/yeast_envs.mut_eff.pdf', dpi=300)    
    

if __name__ == '__main__':
    annotations = pd.read_csv('loci_annotations.csv', index_col=0)
    environments = [line.strip() for line in open('environments.txt')]
    
    params = []
    for environment in environments:
        env_params = []
        for i in range(5):
            print('Environment {}: rep {}'.format(environment, i))
            try:
                p = read_params(environment, id=str(i))
            except FileNotFoundError:
                continue
            env_params.append(p)
        params.append(np.vstack(env_params).mean(0))
    params = pd.DataFrame(params, index=environments)
    manhattan_heatmap(params)
    
    annotations = annotations.join(params.T)
    annotations.to_csv('loci_mutt_eff.csv')
    
    
