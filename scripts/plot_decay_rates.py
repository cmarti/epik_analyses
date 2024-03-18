#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


def plot_decay_factors_heatmap(decay_factors, dataset):
    dataset_labels = {'gb1': 'Protein GB1', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC',
                      'aav': 'AAV2 Capside'}
    figsizes = {'smn1': (4.25, 2.25),
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
    highlight_seq_heatmap(dataset, axes, decay_factors)
    
    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)
    
    # fig.tight_layout()
    fig.savefig('plots/{}.decay_factors.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.decay_factors.pdf'.format(dataset), dpi=300)


def highlight_seq_heatmap(dataset, axes, matrix):
    seqs = {'smn1': 'CAGUAAGU', 
            'gb1': 'VDGV',
            'aav': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'}
    
    axes.set_clip_on(False)
    for x, c in enumerate(seqs[dataset]):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


def read_decay_factors(dataset, id=None, kernel='Rho'):
    if id is None:
        fpath = 'output_gpu/{}.{}.model_params.pth'.format(dataset, kernel)
    else:
        fpath = 'output_gpu/{}.{}.{}.test_pred.csv.model_params.pth'.format(dataset, id, kernel)
    params = torch.load(fpath, map_location=torch.device('cpu'))

    logit_rho = params['covar_module.logit_rho'].numpy()
    log_p = params['covar_module.log_p'].numpy()
    
    rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
    p = np.exp(log_p)
    p = p / np.expand_dims(p.sum(1), 1)
    eta = (1 - p) / p
    
    decay_factor = 1 - (1 - rho) / (1 + eta * rho)
    
    alleles_order = {'gb1': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'aav': ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                             'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'],
                     'smn1': ['A', 'C', 'G', 'U']}
    positions = {'smn1': ['-3', '-2', '-1', '+2', '+3', '+4', '+5', '+6'],
                 'gb1': ['39', '40', '41', '54'],
                 'aav': [str(x) for x in range(561, 589)]}
    
    decay_factor = pd.DataFrame(decay_factor)
    
    if dataset in positions:
        decay_factor.index = positions[dataset]
    
    if dataset in alleles_order:
        decay_factor.columns = sorted(alleles_order[dataset])
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
    

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    # id = '53'
    # for dataset in ['aav', 'gb1', 'smn1']:
    #     decay_factors = read_decay_factors(dataset, id, kernel='ARD')
    #     plot_decay_factors_heatmap(decay_factors, dataset)
    
    annotations = pd.read_csv('loci_annotations.csv', index_col=0)
    annotations['pos'] = np.arange(annotations.shape[0])

    for dataset in ['37C']: # , 'yeast.37C'
        decay_factors = read_decay_factors('yeast.{}'.format(dataset), id='53').mean(1)
        annotations[dataset] = decay_factors
        manhattanplot(dataset, annotations)
        plot_hist(dataset, decay_factors)
