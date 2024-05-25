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

    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    axes.scatter(sel_loci['x'], sel_loci['decay_rate'], c='black', s=7.5, lw=0)

    # Color by chromosome
    ylim = (-0.5, 15)
    axes.grid(alpha=0.1)
    for i in range(0, chr_x.shape[0], 2):
        left, right = chr_x.iloc[i], chr_x.iloc[i+1]
        axes.fill_between(x=(left, right), y1=ylim[0], y2=ylim[1], color='grey', alpha=0.2, lw=0)

    # Get chromosome label positions
    chr_pos = []
    prev_pos = 0
    for pos in chr_x:
        chr_pos.append((pos + prev_pos) / 2)
        prev_pos = pos
    chr_labels = ['chr{}'.format(i+1) for i in range(len(chr_pos))]

    axes.set(title='Growth rate in {}'.format(environment.capitalize()),
             xlabel='Chromomsome position (bp)', ylabel='Decay factor (%)',
             ylim=ylim, xlim=(0, chr_x.max()),
             xticks=chr_pos)
    axes.set_xticklabels(chr_labels, rotation=45)

    np.random.seed(21)
    labels = sel_loci.dropna(subset=['gene']).sort_values('decay_rate', ascending=False).iloc[:10, :]
    dxs = [-1, 1, 1, 2, -1, -1, -0.5, 1, 1.5, -1]
    dys = [0, 1, 0.7, 0, 1, 0, 2, 2, 1, 0]
    for x, y, label, dx, dy in zip(labels['x'], labels['decay_rate'], labels['gene'], dxs, dys):
        xtext = x + 2e5 * dx
        ytext = y + 1 +  dy
        ha = 'left' if xtext > x else 'right'
        axes.annotate(label, xy=(x, y+0.1), ha=ha,
                      xytext=(xtext, ytext), fontsize=7,
                      arrowprops=dict(facecolor='black', shrink=1,
                                      width=0.25, headwidth=2, headlength=2))

    fig.tight_layout()
    fig.savefig('plots/{}.decay_factors.png'.format(environment), dpi=300)
    fig.savefig('plots/{}.decay_factors.pdf'.format(environment), dpi=300)

