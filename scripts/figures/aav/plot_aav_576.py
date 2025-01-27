#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Arial'
    bcs_list = [('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNPVATEQFGSVSTNLQRGNR'), 
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNPVATEQCGSVSTNLQRGNR'),
                ]
    labels_list =[('WT', 'Y576F'),
                  ('WT', 'Y576K'),
                  ]
    nplots = len(bcs_list)
    
    lims = (-11, 5)
    # fig, subplots = plt.subplots(1, nplots, figsize=(3 * nplots, 3))
    fig, subplots = plt.subplots(1, 2, figsize=(2.75 * 2, 2.75 * 1))
    subplots = subplots.flatten()
    # subplots = [subplots]
    ticks = list(range(-10, 5, 2))
    for axes, bcs, labels in zip(subplots, bcs_list, labels_list):
        df1 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bcs[0]), index_col=0)
        df2 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bcs[1]), index_col=0)
        df = df1.join(df2, rsuffix='_2')

        axes.errorbar(df['coef'], df['coef_2'],
                      xerr=2 * df['stderr'],
                      yerr=2 * df['stderr_2'],
                      fmt='', color='grey', alpha=0.25, elinewidth=0.5, lw=0)
        axes.scatter(df['coef'], df['coef_2'], color='black', alpha=0.5, s=3, lw=0, zorder=10)
        axes.axline((0, 0), (1, 1), linestyle='--', lw=0.5, color='grey', alpha=0.5)
        axes.axvline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
        axes.axhline(0, linestyle='--', lw=0.5, color='grey', alpha=0.5)
        
        axes.set(xlabel='Mutational effect in {}'.format(labels[0]),
                ylabel='Mutational effect in {}'.format(labels[1]),
                aspect='equal',
                xlim=lims, ylim=lims, xticks=ticks, yticks=ticks)
        axes.grid(alpha=0.2)
        
    fig.tight_layout(w_pad=2)
    fig.savefig('figures/aav_576_supp.png', dpi=300)
    fig.savefig('figures/aav_576_supp.svg', dpi=300)
    