#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from itertools import combinations
from scipy.stats import mannwhitneyu
from torch.distributions.transforms import CorrCholeskyTransform


def read_decay_factors(dataset, id=None, kernel='Rho', mutation_level=False):
    alleles_order = {'gb1': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'aav': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'smn1': ['A', 'C', 'G', 'U']}
    positions = {'smn1': ['-3', '-2', '-1', '+2', '+3', '+4', '+5', '+6'],
                 'gb1': ['39', '40', '41', '54'],
                 'aav': [str(x) for x in range(561, 589)]}

    if id is None:
        fpath = 'output/{}.{}.test_pred.csv.model_params.pth'.format(dataset, kernel)
    else:
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
                                      for c in corrs])
    else:
        logit_rho = params['covar_module.logit_rho'].numpy()
        log_p = params['covar_module.log_p'].numpy()
        rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
        p = np.exp(log_p)
        p = p / np.expand_dims(p.sum(1), 1)
        eta = (1 - p) / p

        if kernel == 'ARD':
            if mutation_level:
                f = np.sqrt((1 - rho) / (1 + eta * rho))
                f = np.vstack([f[:, i] * f[:, j] for i, j in combinations(np.arange(f.shape[1]), 2)]).T
                decay_factor = 1 - f
        else:
            decay_factor = 1 - (1 - rho) / (1 + eta * rho)

    decay_factor = pd.DataFrame(decay_factor)
    
    if dataset in positions:
        decay_factor.index = positions[dataset]
    
    if dataset in alleles_order:
        alleles = sorted(alleles_order[dataset])

        if kernel == 'GeneralProduct' or mutation_level:
            decay_factor.columns = ['{}-{}'.format(a1, a2) for a1, a2 in combinations(alleles, 2)]
        else:
            decay_factor.columns = alleles
            decay_factor = decay_factor[alleles_order[dataset]]
    
    return(decay_factor)


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    id='53'
    environment = 'li'

    sel_loci = pd.read_csv('raw/qtls.tsv', sep='\t')
    sel_loci = sel_loci.loc[sel_loci['Environment'] == environment, :]
    annotations = pd.read_csv('raw/SNP_list.csv', index_col=0)
    sel_loci = sel_loci.join(annotations, on='SNP_index')
    sel_loci['chr'] = ['chr{}'.format(i) for i in sel_loci['Chromosome']]
    chr_sizes = pd.read_csv('raw/chr_sizes.csv', index_col=0)['size']
    chr_x = np.cumsum(chr_sizes)
    # prev_annot = pd.read_csv('datasets/yeast_annotations.csv', index_col=0).set_index('locus')
    # .join(prev_annot[['gene']], on='locus', rsuffix='_2')
    sel_loci['x'] = chr_x.loc[sel_loci['chr']].values + sel_loci['Position (bp)']
    sel_loci['decay_rate'] = read_decay_factors('qtls_{}'.format(environment), id=id).mean(1).values * 100

    fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
    axes.scatter(sel_loci['Effect'], sel_loci['decay_rate'],
                 s=10, c='black', lw=0, alpha=0.5)
    axes.set(ylabel='Decay rate (%)',
             xlabel='Additive effect',
             title='Growth rate in {}'.format(environment),
             xlim=(-0.15, 0.15))
    axes.grid(alpha=0.2)
    
    fig.tight_layout()
    fig.savefig('plots/qtls_decay_rates_vs_additive_scatter_{}.png'.format(environment), dpi=300)
    fig.savefig('plots/qtls_decay_rates_vs_additive_scatter_{}.pdf'.format(environment))
