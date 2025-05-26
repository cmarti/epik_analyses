#!/usr/bin/env python
import sys
import numpy as np
import mavenn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
from itertools import combinations


def plot_theta_heatmap(theta, dataset, wts):
    figsizes = {'smn1': (4, 2),
                'gb1': (2.5, 5),
                'aav': (7, 5)}
    
    fig, axes = plt.subplots(1, 1, figsize=figsizes[dataset])
    
    if dataset == 'gb1':
        cbar_axes = fig.add_axes([0.65, 0.25, 0.04, 0.5])
        fig.subplots_adjust(right=0.60, left=0.2)
    elif dataset == 'aav':
        cbar_axes = fig.add_axes([0.85, 0.275, 0.02, 0.5])
        fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    else:
        cbar_axes = fig.add_axes([0.85, 0.275, 0.03, 0.5])
        fig.subplots_adjust(right=0.82, left=0.12, bottom=0.2)
    
    cmap = cm.get_cmap('binary')
    axes.set_facecolor(cmap(0.1))
    vmax = min(3, np.ceil(np.abs(theta.values).max()))

    sns.heatmap(theta.T, ax=axes, cmap='coolwarm', 
                vmax=3, vmin=-vmax, center=0,
                cbar_ax=cbar_axes,
                cbar_kws={'label': r'Mutational effect'})
    axes.set(xlabel='Position', ylabel='Allele')
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(dataset, axes, theta, wts)
    
    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)
    
    # fig.tight_layout()
    fig.savefig('figures/{}.theta_heatmap.png'.format(dataset), dpi=300)
    fig.savefig('figures/{}.theta_heatmap.pdf'.format(dataset), dpi=300)


def read_theta(dataset, wts):
    aa_order = ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C']
    positions = {'smn1': ['-3', '-2', '-1', '+2', '+3', '+4', '+5', '+6'],
                 'gb1': ['39', '40', '41', '54'],
                 'aav': [str(x) for x in range(561, 589)]}
    wt = wts[dataset]
    
    # Load model
    fpath = 'results/{}.global_epistasis.model'.format(dataset)
    model = mavenn.load(fpath)

    # Extract parameters
    theta = model.get_theta(gauge='uniform')['logomaker_df']

    # Find best sequence
    seq0 = ''.join([theta.columns[i] for i in np.argmax(theta, axis=1)])
    phi_max = model.x_to_phi(seq0)

    # Normalize thetas so that the average absolute mutational effect is 1
    mut_effs = np.hstack([theta[i] - theta[j]
                          for i, j in combinations(theta.columns, 2)])
    sd = np.nanmean(np.abs(mut_effs))
    theta = theta / sd
    wt_theta = np.array([[theta[a].iloc[p]] for p, a in enumerate(wt)])
    theta = theta - wt_theta
    
    # Name sites
    if dataset in positions:
        theta.index = positions[dataset]

    # Order alleles
    if dataset in ['gb1', 'aav']:
        theta = theta[aa_order]

    # Load data
    fpath = 'datasets/{}.csv'.format(dataset)
    data = pd.read_csv(fpath, index_col=0)

    # Calculate phi on the data
    phi = model.x_to_phi(data.index.values)
    data['phi'] = (phi - phi_max) / sd

    # Compute predicted non-linearity
    phi = np.linspace(phi.min() - 1, phi_max + 1, 101)
    yhat = model.phi_to_yhat(phi)
    phi = (phi - phi_max) / sd
    pred = pd.DataFrame({'phi': phi, 'yhat': yhat})

    return(theta, pred, data.dropna())


def highlight_seq_heatmap(dataset, axes, matrix, wts):
    axes.set_clip_on(False)
    for x, c in enumerate(wts[dataset]):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(patches.Rectangle(xy=(x, y), width=1., height=1., lw=0.75, fill=False, edgecolor='black', zorder=2))


def plot_nonlinearity(pred, data, dataset):
    ylabels = {'smn1': 'Percent Spliced In (%)',
               'gb1': 'log(Enrichment)',
               'aav': 'log(Enrichment)'}

    fig, axes = plt.subplots(1, 1, figsize=(4., 2.75))

    H, xbins, ybins = np.histogram2d(x=data['phi'].values, 
                                     y=data['y'].values, bins=100)
    dy = (ybins[-1] - ybins[0])
    aspect = (xbins[-1] - xbins[0]) / dy
    log_freqs = np.log10(H.T[::-1, :]+1)
    im = axes.imshow(log_freqs, cmap='Greys',
                     extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
                     aspect=aspect, vmax=np.log10(1001))
    ticks = [0, np.log10(2), np.log10(11), np.log10(101), np.log10(1001)]
    plt.colorbar(im, shrink=0.9, ticks=ticks, label='Number of genotypes')
    fig.axes[-1].set_yticklabels(['0', '1', '10', '$10^2$', '$10^3$'])

    axes.plot(pred['phi'], pred['yhat'], c='black', lw=0.75, linestyle='--')
    ylim = (ybins[0] - 0.05 * dy, ybins[-1] + 0.05 * dy)
    axes.set(ylim=ylim, xlim=(pred['phi'].min(), pred['phi'].max()),
             xlabel=r'Latent phenotype $\phi$',
             ylabel=ylabels[dataset])
    axes.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig('figures/{}.global_epistasis.png'.format(dataset), dpi=300)
    fig.savefig('figures/{}.global_epistasis.pdf'.format(dataset), dpi=300)


if __name__ == '__main__':
    wts = {'smn1': 'CAGUAAGU', 
           'gb1': 'VDGV',
           'aav': 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'}

    for dataset in ['smn1', 'gb1', 'aav']:
        theta, pred, data = read_theta(dataset, wts)
        plot_theta_heatmap(theta, dataset, wts)
        plot_nonlinearity(pred, data, dataset)