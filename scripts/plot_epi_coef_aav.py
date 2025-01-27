#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns


    

if __name__ == '__main__':
    bc = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    data = pd.read_csv('output/aav.Jenga.{}_expansion.csv'.format(bc), index_col=0)
    data = data.loc[['_' in x for x in data.index], :]
    data['mut1'] = [x.split('_')[0] for x in data.index]
    data['mut2'] = [x.split('_')[1] for x in data.index]
    data['pos1'] = [int(x[1:-1]) for x in data['mut1']]
    data['pos2'] = [int(x[1:-1]) for x in data['mut2']]
    data['sq_coef']  = data['coef'] ** 2

    print(data.loc[data['pos2'] == 27, :].sort_values('coef', ascending=False).head(20))
    print(data.loc[data['pos2'] == 27, :].sort_values('coef', ascending=True).head(20))

    print(data.loc[data['mut2'] == 'R27K', :].sort_values('coef', ascending=False).head(20))
    exit()

    df = pd.pivot_table(data, index='pos1', columns='pos2', values='sq_coef', aggfunc='mean')
    cols = np.arange(28)
    df = df.reindex(cols).T.reindex(cols).T.fillna(0.)
    df = df + df.T
    positions = np.arange(561, 561+28)

    fig, axes = plt.subplots(1, 1, figsize=(4.75, 4))
        
    sns.heatmap(df, ax=axes, cmap='binary',
                cbar_kws={'label': 'Average $\epsilon^2$', 'shrink': 0.8})
    axes.set(xlabel='Position 1', ylabel='Position 2',
             xticks=np.arange(0, 28)+0.5, yticks=np.arange(0, 28)+0.5)
    axes.set_xticklabels(positions, rotation=90, fontsize=8)
    axes.set_yticklabels(positions, rotation=0, fontsize=8)
    
    sns.despine(right=False, top=False)

    fig.tight_layout()
    fig.savefig('plots/aav_epistatic_coef.png', dpi=300)
    