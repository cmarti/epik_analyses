#!/usr/bin/env python
import sys
import numpy as np
import mavenn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


def plot_theta_heatmap(theta, dataset):
    dataset_labels = {'gb1': 'Protein GB1', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC',
                      'aav': 'AAV2 Capside'}
    figsizes = {'smn1': (4, 2.25),
                'gb1': (3, 6),
                'aav': (8, 6)}
    
    fig, axes = plt.subplots(1, 1, figsize=figsizes[dataset])
    
    if dataset == 'gb1':
        cbar_axes = fig.add_axes([0.65, 0.25, 0.04, 0.5])
        fig.subplots_adjust(right=0.60, left=0.2)
    else:
        cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
        fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(theta.T, ax=axes, cmap='Blues', 
                vmax=0,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'Additive contribution'})
    
    axes.set(title=dataset_labels[dataset],
             xlabel='Position', ylabel='Allele')
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(dataset, axes, theta)
    
    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)
    
    # fig.tight_layout()
    fig.savefig('plots/{}.theta.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.theta.pdf'.format(dataset), dpi=300)


def read_theta(dataset, id):
    aa_order = ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C']
    positions = {'smn1': ['-3', '-2', '-1', '+2', '+3', '+4', '+5', '+6'],
                 'gb1': ['39', '40', '41', '54'],
                 'aav': [str(x) for x in range(561, 589)]}
    fpath = 'output_gpu/{}.{}.mavenn.params.csv'.format(dataset, id)
    theta = pd.read_csv(fpath, index_col=0)
    vmax = theta.max(axis=1).values.reshape((theta.shape[0], 1))
    theta = theta - vmax
    sd = np.abs(np.nanmean(theta.values) * theta.shape[1]/ (theta.shape[1] - 1))
    theta = theta / sd
    
    if dataset in positions:
        theta.index = positions[dataset]
    if dataset in ['gb1', 'aav']:
        theta = theta[aa_order]

    fpath = 'output_gpu/{}.{}.mavenn.model'.format(dataset, id)
    model = mavenn.load(fpath)
    seq0 = ''.join([theta.columns[i] for i in np.where(theta == 0.)[1]])
    phi_max = model.x_to_phi(seq0)

    fpath = 'datasets/{}.csv'.format(dataset)
    data = pd.read_csv(fpath, index_col=0)
    data['phi'] = model.x_to_phi(data.index.values)

    phi = np.linspace(data['phi'].min() - 1, phi_max + 1, 101)
    yhat = model.phi_to_yhat(phi)
    phi = (phi - phi_max) / sd
    pred = pd.DataFrame({'phi': phi, 'yhat': yhat})
    data['phi'] = (data['phi'] - phi_max) / sd
    return(theta, pred, data.dropna())


def highlight_seq_heatmap(dataset, axes, matrix):
    seqs = {'smn1': 'CAGUAAGU', 
            'gb1': 'VDGV',
            'aav': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'}
    
    axes.set_clip_on(False)
    for x, c in enumerate(seqs[dataset]):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


def manhattanplot(dataset, values):
    dataset_labels = {'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC'}
    fig, axes = plt.subplots(1, 1, figsize=(8, 2.5))
    xs = np.arange(values.shape[0])
    axes.scatter(xs, values, c='black', s=5, alpha=0.7, lw=0)
    axes.set(title=dataset_labels[dataset],
             xlabel='Locus',
             ylabel='Additive contribution',
             xlim=(-1, values.shape[0] + 1),
             xticks=[])
    sns.despine(ax=axes, right=False, top=False)
    axes.grid(alpha=0.2)
    
    fig.tight_layout()
    fig.savefig('plots/{}.theta.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.theta.pdf'.format(dataset), dpi=300)
    

def plot_hist_theta(dataset, values):
    dataset_labels = {'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC'}
    
    fig, axes = plt.subplots(1, 1, figsize=(3, 2.5))
    sns.histplot(values, ax=axes, bins=30)
    axes.set(title=dataset_labels[dataset],
             xlabel='Latent mutational effects',
             ylabel='# mutations', xlim=(None, 0))
    sns.despine(ax=axes, right=False, top=False)
    
    fig.tight_layout()
    fig.savefig('plots/{}.theta_hist.png'.format(dataset), dpi=300)
    fig.savefig('plots/{}.theta_hist.pdf'.format(dataset), dpi=300)


def plot_nonlinearity(pred, data, dataset):
    fig, axes = plt.subplots(1, 1, figsize=(4., 3.))

    H, xbins, ybins = np.histogram2d(x=data['phi'].values, 
                                     y=data['y'].values, bins=100)
    dy = (ybins[-1] - ybins[0])
    aspect = (xbins[-1] - xbins[0]) / dy
    axes.imshow(np.log(H.T[::-1, :]), cmap='Blues',
                extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
                aspect=aspect)
    axes.plot(pred['phi'], pred['yhat'], c='black')
    ylim = (ybins[0] - 0.05 * dy, ybins[-1] + 0.05 * dy)
    axes.set(ylim=ylim, xlim=(data['phi'].min() - 1, 1),
             xlabel=r'Latent phenotype $\phi$',
             ylabel=r'Predicted phenotype $\hat{y}$')
    axes.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('plots/{}.nonlinearity.png'.format(dataset), dpi=300)


if __name__ == '__main__':
    id = '53'
    
    for dataset in ['smn1', 'gb1', 'aav']:
        theta, pred, data = read_theta(dataset, id)
        plot_theta_heatmap(theta, dataset)
        plot_nonlinearity(pred, data, dataset)
    
    for dataset in ['yeast', 'yeast.37C']:
        theta, pred, data = read_theta(dataset, id)
        theta = theta.values.min(1).flatten()
        manhattanplot(dataset, theta)
        plot_nonlinearity(pred, data, dataset)
        # plot_hist_theta(dataset, theta)
