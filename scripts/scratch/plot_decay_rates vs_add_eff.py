#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from os.path import exists


def calc_decay_factors(params):
    logit_rho = params['covar_module.logit_rho'].numpy()
    log_p = params['covar_module.log_p'].numpy()
    rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
    p = np.exp(log_p)
    p = p / np.expand_dims(p.sum(1), 1)
    eta = (1 - p) / p
    decay_factor = 1 - (1 - rho) / (1 + eta * rho)
    return(decay_factor)


def read_decay_factors(dataset, kernel='Rho', subsamples=False):
    if subsamples:
        decay_factors = []
        for i in range(5):
            fpath = 'output/{}.{}.{}.test_pred.csv.model_params.pth'.format(dataset, i, kernel)
            if not exists(fpath):
                continue
            params = torch.load(fpath, map_location=torch.device('cpu'))
            f = calc_decay_factors(params).mean(1)
            decay_factors.append(f)
        decay_factors = np.vstack(decay_factors).T
    else:
        fpath = 'output/yeast.{}.{}.model_params.pth'.format(dataset, kernel)
        params = torch.load(fpath, map_location=torch.device('cpu'))
        decay_factors = calc_decay_factors(params).T
    return(decay_factors)


if __name__ == '__main__':
    environment = '37C'
    kernel = 'Rho'

    annotations = pd.read_csv('datasets/yeast_annotations.csv', index_col=0)
    annotations['pos'] = np.arange(annotations.shape[0])
    f = read_decay_factors(environment, kernel=kernel, subsamples=True).mean(1)
    annotations['decay_rate'] = f

    values = np.load('datasets/l21_linear_effects.npy')
    environments = [line.strip() for line in open('environments.txt')]
    params = pd.DataFrame(values, index=sorted(environments)).loc[environments, :]
    annotations['add_eff'] = params.loc[environment, :].values

    
    fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
    axes.scatter(annotations['add_eff'], annotations['decay_rate'] * 100,
                 s=10, c='black', lw=0, alpha=0.5)
    axes.set(ylabel='Decay rate (%)',
             xlabel='Additive effect',
             title='Growth rate in {}'.format(environment),
             xlim=(-0.07, 0.07))
    axes.grid(alpha=0.2)
    
    fig.tight_layout()
    fig.savefig('plots/decay_rates_vs_additive_scatter_{}.png'.format(environment), dpi=300)
    fig.savefig('plots/decay_rates_vs_additive_scatter_{}.pdf'.format(environment))
