#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    bcs_list = [('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNPVATEQFGSVSTNLQRGNR'), 
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTRPVATEQYGSVSTNLQRGNR'),
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTVNPVATEQYGSVSTNLQRGNR'),
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNAVATEQYGSVSTNLQRGNR'),
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNPVATEQYGSVSTNLQQGNR'),
                ('DEEEIRTTNPVATEQYGSVSTNLQRGNR', 'DEEEIRTTNPVATEQYGSVSTNLQRGER'),
                ]
    labels_list =[('WT', 'Y576F'),
                  ('WT', 'N569R'),
                  ('WT', 'T568V'),
                  ('WT', 'P570A'),
                  ('WT', 'R585Q'),
                  ('WT', 'N587E')
                  ]
    nplots = len(bcs_list)
    
    lims = (-11, 5)
    # fig, subplots = plt.subplots(1, nplots, figsize=(3 * nplots, 3))
    fig, subplots = plt.subplots(3, 2, figsize=(3 * 2, 3 * 3))
    subplots = subplots.flatten()
    # subplots = [subplots]
    ticks = list(range(-10, 5, 2))
    for axes, bcs, labels in zip(subplots, bcs_list, labels_list):
        df1 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bcs[0]), index_col=0)
        mut_label = labels[1][0] + str(int(labels[1][1:-1])-561) + labels[1][-1]
        mut_eff = df1.loc[mut_label, 'coef']
        df2 = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bcs[1]), index_col=0)
        data = df1.join(df2, rsuffix='_2')

        idx = (data['coef'] > 0.75) & (data['coef_2'] < -0.75)
        if idx.sum() > 0:
            print('Sign epistasis with {}'.format(labels[1]))
            print(data.loc[idx, :])

        # print(data.sort_values('lower_ci_2', ascending=False).head(20))
        df = data#.loc[data['even'] == 0, :]
        axes.scatter(df['coef'], df['coef_2'], s=5, lw=0, c='black')
        # axes.errorbar(df['coef'], df['coef_2'],
        #             xerr=2 * df['stderr'],
        #             yerr=2 * df['stderr_2'],
        #             fmt='', color='black', alpha=0.25, markersize=3, elinewidth=0.5, lw=0)
        axes.axline((0, 0), (1, 1), linestyle='--', lw=0.75, color='grey')
        axes.axvline(0, linestyle='--', lw=0.75, color='grey')
        axes.axhline(0, linestyle='--', lw=0.75, color='grey')
        axes.axvline(mut_eff, linestyle='--', lw=0.75, color='red')
        
        axes.set(xlabel='Mutational effect in {}'.format(labels[0]),
                ylabel='Mutational effect in {}'.format(labels[1]),
                aspect='equal',
                xlim=lims, ylim=lims, xticks=ticks, yticks=ticks)
        axes.grid(alpha=0.2)
        
    fig.tight_layout()
    fig.savefig('plots/aav_mut_effs_bcs.png', dpi=300)
    