#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
import logomaker

    

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    
    wt = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    wt_bc = 'DEEEIRTTNPVATEQGSVSTNLQRGNR'
    wt_bc2 = 'DEEEIRTTPVATEQYGSVSTNLQRGNR'
    ref = 561
    charge = {'E': -1, 'D': -1, 'R': 1, 'K': 1}
    
    data = pd.read_csv('datasets/aav.csv', index_col=0)
    lims = data['y'].min(), data['y'].max()
    data['576'] = [x[576-ref] for x in data.index]
    data['576_other'] = [x[:576-ref] + x[576-ref+1:] for x in data.index]
    data['569'] = [x[569-ref] for x in data.index]
    data['569_other'] = [x[:569-ref] + x[569-ref+1:] for x in data.index]
    data['charge'] = [np.sum([charge.get(aa, 0) for aa in seq[-10:]]) for seq in data.index.values]
    data['d_to_wt'] = [np.sum([a1 != a2 for a1, a2 in zip(seq, wt)]) for seq in data.index.values]
    data['charge_2'] = data['charge'] == 2
    
    idx = (data['y'] >= 0) & (~np.isin(data['576'], ['W', 'F', 'Y']))
    df2 = data.loc[idx, :]
    m = logomaker.alignment_to_matrix(df2.index.values, to_type='probability', pseudocount=0)
    pos = np.arange(561, 588)
    print(df2)
    print(data.loc[(data['y'] >= 0) & (np.isin(data['576'], ['W', 'F', 'Y'])), ['d_to_wt', 'charge', 'charge_2']].mean())
    print(df2[['d_to_wt', 'charge', 'charge_2']].mean())
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 1.5))
    logomaker.Logo(m, ax=axes, color_scheme='chemistry')
    axes.set(xlabel='Position', ylabel='Frequency', xticks=np.arange(27))
    axes.set_xticklabels(pos, rotation=90)
    fig.tight_layout()
    fig.savefig('plots/compensation_hydrophobics.png', dpi=300)

    df = pd.pivot_table(data, index='576_other', columns='576', values='y')
    df['n'] = np.isnan(df).sum(1)
    df = df.loc[df['n']<19, :].dropna(subset=['Y'])
    df3 = df.loc[[wt_bc], :]
    print(df)

    fig, subplots = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ticks = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5]

    axes = subplots[0]
    axes.scatter(df['Y'], df['F'], color='black', s=5, lw=0, alpha=0.6)
    axes.scatter(df3['Y'], df3['F'], color='red', s=5, lw=0, alpha=1)
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), lw=0.75, c='grey', linestyle='--')
    axes.axvline(0, lw=0.75, c='grey', linestyle='--')
    axes.axhline(0, lw=0.75, c='grey', linestyle='--')
    axes.set(xlabel='576Y DMS score', ylabel='576F DMS score',
             xlim=lims, ylim=lims, aspect='equal', xticks=ticks, yticks=ticks)
    
    df = pd.melt(df.drop('n', axis=1).reset_index(), id_vars=['Y', 'F', 'W', '576_other'])
    df.columns = ['Y', 'F', 'W', '576_other', 'aa1', 'v1']
    df = pd.melt(df, id_vars=['aa1', 'v1', '576_other']).dropna()
    df3 = df.loc[df['576_other'] == wt_bc, :]

    axes = subplots[1]
    axes.scatter(df['value'], df['v1'], color='black', s=5, lw=0, alpha=0.6)
    axes.scatter(df3['value'], df3['v1'], color='red', s=5, lw=0, alpha=1)
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), lw=0.75, c='grey', linestyle='--')
    axes.axvline(0, lw=0.75, c='grey', linestyle='--')
    axes.axhline(0, lw=0.75, c='grey', linestyle='--')
    axes.set(xlabel='576-Y/F/W DMS score', ylabel='576-Other DMS score',
             xlim=lims, ylim=lims, aspect='equal', xticks=ticks, yticks=ticks)

    df = pd.pivot_table(data, index='569_other', columns='569', values='y')
    df['n'] = np.isnan(df).sum(1)
    df = df.loc[df['n']<19, :].dropna(subset=['N'])
    df = pd.melt(df.drop('n', axis=1).reset_index(), id_vars=['N', '569_other']).dropna()
    print(df)
    df3 = df.loc[df['569_other'] == wt_bc2, :]

    axes = subplots[2]
    axes.scatter(df['N'], df['value'], color='black', s=5, lw=0, alpha=0.6)
    axes.scatter(df3['N'], df3['value'], color='red', s=5, lw=0, alpha=1)
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), lw=0.75, c='grey', linestyle='--')
    axes.axvline(0, lw=0.75, c='grey', linestyle='--')
    axes.axhline(0, lw=0.75, c='grey', linestyle='--')
    axes.set(xlabel='569-N DMS score', ylabel='569-Other DMS score',
             xlim=lims, ylim=lims, aspect='equal', xticks=ticks, yticks=ticks)

    fig.tight_layout()
    fig.savefig('plots/aav_validations_2.png', dpi=300)


    bins = np.linspace(lims[0], lims[1], 50)
    fig, subplots = plt.subplots(3, 1, figsize=(2.75, 7.5), gridspec_kw={'hspace': 0.5})

    axes = subplots[0]
    xticks = np.array([-10, -5, 0, 5, 10])
    axes.hist(data.loc[data['569'] != 'N', 'y'], color='grey', label='Other',
              alpha=0.5, bins=bins, density=True)
    axes.hist(data.loc[data['569'] == 'N', 'y'], color='black', label='569N',
              alpha=0.5, bins=bins, density=True)
    axes.legend(loc=0, fontsize=9)
    axes.grid(alpha=0.2)
    axes.set(xlabel='DMS score', ylabel='Probability density', xticks=xticks)

    axes = subplots[1]
    idx = np.isin(data['576'], ['Y', 'F', 'W'])
    axes.hist(data.loc[~idx, 'y'], color='grey', label='Other',
              alpha=0.5, bins=bins, density=True)
    axes.hist(data.loc[idx, 'y'], color='black', label='576Y/F/W',
              alpha=0.5, bins=bins, density=True)
    axes.legend(loc=0, fontsize=9)
    axes.grid(alpha=0.2)
    axes.set(xlabel='DMS score', ylabel='Probability density', xticks=xticks)

    axes = subplots[2]
    sns.violinplot(y='y', x='charge', data=data, color='grey', inner=None, linewidth=0.75)
    axes.set(ylabel='DMS score', xlabel='579-588 charge')
    axes.grid(alpha=0.2)
    # axes.scatter(data['charge'], data['y'], s=5, alpha=0.2, c='black', lw=0)

    # fig.tight_layout()
    fig.savefig('plots/aav_validations.png', dpi=300)
    fig.savefig('plots/aav_validations.svg', dpi=300)