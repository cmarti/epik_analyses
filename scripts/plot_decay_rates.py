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


def plot_decay_factors_heatmap(decay_factors, dataset):
    dataset_labels = {'gb1': 'Protein GB1', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC',
                      'aav': 'AAV2 Capside'}
    figsizes = {'smn1': (4.5, 2.25),
                'gb1': (3, 6),
                'aav': (8, 6)}
    
    fig, axes = plt.subplots(1, 1, figsize=figsizes[dataset])
    
    if dataset == 'gb1':
        cbar_axes = fig.add_axes([0.65, 0.25, 0.04, 0.5])
        fig.subplots_adjust(right=0.60, left=0.2)
    else:
        cbar_axes = fig.add_axes([0.80, 0.275, 0.03, 0.5])
        fig.subplots_adjust(right=0.77, left=0.12, bottom=0.2)
    
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(decay_factors.T * 100, ax=axes, cmap='Blues', 
                vmin=0, vmax=100,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'Decay factor (%)'})
    
    axes.set(title=dataset_labels[dataset],
             xlabel='Position', ylabel='Allele')
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    # highlight_seq_heatmap(dataset, axes, decay_factors)
    
    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)
    return(fig)


def highlight_seq_heatmap(dataset, axes, matrix):
    seqs = {'smn1': 'CAGUAAGU', 
            'gb1': 'VDGV',
            'aav': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'}
    
    axes.set_clip_on(False)
    for x, c in enumerate(seqs[dataset]):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


def read_decay_factors(dataset, id=None, kernel='Rho', mutation_level=False):
    alleles_order = {'gb1': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'aav': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'smn1': ['A', 'C', 'G', 'U']}
    positions = {'smn1': ['-3', '-2', '-1', '+1', '+2', '+3', '+4', '+5', '+6'],
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

        if kernel == 'ARD' and mutation_level:
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


def manhattanplot(dataset, df):
    dataset_labels = {'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC'}
    fig, axes = plt.subplots(1, 1, figsize=(8, 3))

    axes.scatter(df.loc[df['even'], 'pos'], df.loc[df['even'], dataset], c='black', s=5, alpha=0.7, lw=0)
    axes.scatter(df.loc[~df['even'], 'pos'], df.loc[~df['even'], dataset], c='grey', s=5, alpha=0.7, lw=0)

    chroms = annotations.groupby(['chr'])['pos'].mean()
    chroms_bounds = annotations.groupby(['chr'])['pos'].max()

    ylim = (5e-6, 1)
    axes.vlines(chroms_bounds.values + 0.5, ymin=ylim[0], ymax=ylim[1], lw=0.5, linestyles='--', colors='grey')
    axes.set(title=dataset_labels.get(dataset, dataset),
             xlabel='Locus', ylabel='Decay factor',
             xlim=(-1, df[dataset].shape[0] + 1),
             xticks=chroms.values,
             yscale='log', ylim=ylim)
    axes.set_xticklabels(chroms.index, rotation=45)

    labels = annotations.dropna(subset=['gene'])
    for x, y, label in zip(labels['pos'], labels[dataset], labels['gene']):
        axes.text(x, y, label, fontsize=6)

    sns.despine(ax=axes, right=False, top=False)
    
    fig.tight_layout()
    fig.savefig('plots/{}.decay_factors.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.decay_factors.pdf'.format(dataset), dpi=300)


def plot_hist(dataset, values):
    dataset_labels = {'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC'}
    
    fig, axes = plt.subplots(1, 1, figsize=(3, 2.5))
    sns.histplot(np.log(values), ax=axes, bins=30)
    axes.set(title=dataset_labels.get(dataset, dataset),
             xlabel='log(Decay factor)',
             ylabel='# loci')
    sns.despine(ax=axes, right=False, top=False)
    
    fig.tight_layout()
    fig.savefig('plots/{}.decay_factors_hist.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.decay_factors_hist.pdf'.format(dataset), dpi=300)
    

def calc_mut_decay_rates_ARD(decay_rates):
    decay_factors = 1 - decay_rates
    mut = {}
    for a1, a2 in combinations(decay_factors.columns, 2):
        mut['{}-{}'.format(a1, a2)] = 1 - decay_factors[a1] * decay_factors[a2]
    mut = pd.DataFrame(mut)
    return(mut)
    
    
def plot_aav_mut_decay_rates_distrib(decay_rates):
    subsets = {'Buried': np.arange(561, 574).astype(str),
               'Interface': np.arange(576, 581).astype(str),
               'Surface': np.arange(583, 589).astype(str)}
    fig, subplots = plt.subplots(3, 1, figsize=(3, 4), sharex=True, sharey=True)
    bins = np.linspace(-1, 2, 50)
    for axes, (label, sites) in zip(subplots, subsets.items()):
        x = decay_rates.loc[sites, :].values.flatten()
        sns.histplot(np.log10(x * 100), ax=axes,
                     alpha=0.3, color='grey',
                     bins=bins, stat='percent')
        axes.set(ylabel='% mutations')
        axes.text(0.025, 0.9, label, transform=axes.transAxes,
                  ha='left', va='top')
    axes.set(xlabel='Mutation decay rate ($log_{10}$%)')
    fig.tight_layout()
    fig.savefig('plots/aav.mutation_decay_rates.png', dpi=300)
    
    for s1, s2 in combinations(subsets, 2):
        x1 = decay_rates.loc[subsets[s1], :].values.flatten()
        x2 = decay_rates.loc[subsets[s2], :].values.flatten()
        print(s1, s2, mannwhitneyu(x1, x2))


if __name__ == '__main__':
    kernel = 'ARD'
    plt.rcParams['font.family'] = 'Arial'
    id='53'

    for dataset in ['aav']:
        # decay_factors = read_decay_factors(dataset, id=id, kernel='Rho', mutation_level=False).mean(1)
        # decay_factors.to_csv('results/{}.Rho.decay_factors.csv'.format(dataset))
        # exit()
        
        decay_factors = read_decay_factors(dataset, kernel=kernel, mutation_level=False)
        decay_factors.to_csv('results/{}.decay_factors.csv'.format(dataset))
        print(decay_factors)
        fig = plot_decay_factors_heatmap(decay_factors, dataset)

        fig.savefig('plots/{}.{}.decay_factors.png'.format(dataset, kernel), dpi=300)
        fig.savefig('plots/{}.{}.decay_factors.pdf'.format(dataset, kernel), dpi=300)
    exit()
    

    # mut_decay_rates = calc_mut_decay_rates_ARD(decay_factors)
    # plot_aav_mut_decay_rates_distrib(mut_decay_rates)
    
    annotations = pd.read_csv('datasets/yeast_annotations.csv', index_col=0)
    annotations['pos'] = np.arange(annotations.shape[0])

    for dataset in ['li']: # , 'yeast.37C'
        decay_factors = read_decay_factors('yeast_{}'.format(dataset), id=id).mean(1)
        annotations[dataset] = decay_factors
        manhattanplot(dataset, annotations)
        plot_hist(dataset, decay_factors)
