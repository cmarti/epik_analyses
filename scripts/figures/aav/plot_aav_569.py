#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    fig, subplots = plt.subplots(1, 2, figsize=(2.75 * 2, 2.75 * 1))
    
    axes = subplots[0]
    wt = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    mut = 'DEEEIRTTQPVATEQYGSVSTNLQRGNR'
    df1 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(wt), index_col=0)
    df2 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(mut), index_col=0)
    print(df1)
    print(df2)
    
    data = df1.join(df2, rsuffix='_2').dropna()
    lims = (-11, 5)
    ticks = list(range(-10, 5, 2))
    df = data
    axes.errorbar(df['coef'], df['coef_2'],
                  xerr=2 * df['stderr'],
                  yerr=2 * df['stderr_2'],
                  fmt='', color='grey', alpha=0.25, elinewidth=0.5, lw=0)
    axes.scatter(df['coef'], df['coef_2'], color='black', alpha=0.5, s=3, lw=0, zorder=10)
    axes.axline((0, 0), (1, 1), linestyle='--', lw=0.5, color='grey', alpha=0.5)
    axes.axvline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
    axes.axhline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
    
    axes.set(xlabel='Mutational effect in WT',
             ylabel='Mutational effect in N569Q',
             aspect='equal', xlim=lims, ylim=lims,
             xticks=ticks, yticks=ticks)
    axes.grid(alpha=0.2)
    
    
    axes = subplots[1]
    ref = 561
    data = pd.read_csv('datasets/aav.csv', index_col=0)
    lims = data['y'].min(), data['y'].max()
    data['569'] = [x[569-ref] for x in data.index]
    data['569_other'] = [x[:569-ref] + x[569-ref+1:] for x in data.index]

    ticks = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5]
    df = pd.pivot_table(data, index='569_other', columns='569', values='y')
    df['n'] = np.isnan(df).sum(1)
    df = df.loc[df['n']<19, :].dropna(subset=['N'])
    df = pd.melt(df.drop('n', axis=1).reset_index(), id_vars=['N', '569_other']).dropna()

    axes.scatter(df['N'], df['value'], color='black', alpha=0.75, s=3, lw=0)
    axes.grid(alpha=0.2)
    axes.axline((0, 0), (1, 1), linestyle='--', lw=0.5, color='grey', alpha=0.5)
    axes.axvline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
    axes.axhline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
    axes.set(xlabel='569-Asn DMS score', ylabel='569-Other DMS score',
             xlim=lims, ylim=lims, aspect='equal', xticks=ticks, yticks=ticks)
    
    fig.tight_layout(w_pad=2)
    fig.savefig('figures/aav_569_supp.png', dpi=300)
    fig.savefig('figures/aav_569_supp.svg', dpi=300)
    
    
    # Compensatory mutations
    ref = 561
    charge = {'E': -1, 'D': -1, 'R': 1, 'K': 1}
    
    data = pd.read_csv('datasets/aav.csv', index_col=0)
    lims = data['y'].min(), data['y'].max()
    data['569'] = [x[569-ref] for x in data.index]
    data['576'] = [x[576-ref] for x in data.index]
    data['576_other'] = [x[:576-ref] + x[576-ref+1:] for x in data.index]
    data['588'] = [x[588-ref] for x in data.index]
    data['588_other'] = [x[:588-ref] + x[588-ref+1:] for x in data.index]
    data['585'] = [x[588-ref] for x in data.index]
    data['585_other'] = [x[:585-ref] + x[585-ref+1:-1] for x in data.index]

    bins = np.linspace(lims[0], lims[1], 50)
    fig, axes = plt.subplots(1, 1, figsize=(2.75, 3.5))

    xticks = np.array([-10, -5, 0, 5, 10])
    
    aromatics = np.isin(data['576'], ['Y', 'F', 'W'])
    compensatory = (~np.isin(data['585'], ['K', 'R'])) | (~np.isin(data['588'], ['K', 'R']))
    
    idx = ~aromatics
    axes.hist(data.loc[idx, 'y'], color='grey', label='Other',
              alpha=0.5, bins=bins, density=True)
    
    idx = aromatics
    axes.hist(data.loc[idx, 'y'], color='black', label='576F/Y/W',
              alpha=0.5, bins=bins, density=True)
    
    x = data.loc['DEEEIRTTNPVATEQCGSVSTNLQRGNL', 'y']
    axes.axvline(x, color='purple', label='Other+R588L', lw=0.75, linestyle='--')
    
    x = data.loc['DEEEIRTTNPVATEQCGSVSTNLQHGNA', 'y']
    axes.axvline(x, color='darkred', label='Other+R585H/R588A', lw=0.75, linestyle='--')
    
    
    axes.legend(loc=(0.1, 1.05), fontsize=9)
    axes.grid(alpha=0.2)
    axes.set(xlabel='DMS score', ylabel='Probability density', xticks=xticks)
    
    fig.tight_layout(pad=0.5)
    fig.savefig('figures/aav_576_compensation.png', dpi=300)
    fig.savefig('figures/aav_576_compensation.svg', dpi=300)
    