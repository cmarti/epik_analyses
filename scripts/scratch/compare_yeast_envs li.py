#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


if __name__ == '__main__':
    subset = 'qtls'
    environments = ['30C', 'li']

    df = pd.read_csv('datasets/{}_all_envs.csv'.format(subset), index_col=0)

    pairs = [('30C', '27C'), ('30C', '33C'), ('30C', '37C'),
             ('30C', 'cu'), ('30C', 'sds'), ('30C', 'eth'),
             ('30C', '4NQO'), ('30C', 'gu'), ('30C', 'li')]
    
    fig, subplots = plt.subplots(3, 3, figsize=(9, 8), sharex=True, sharey=True)    
    subplots = subplots.flatten()

    xmin, xmax = np.nanmin(df.values), np.nanmax(df.values)
    lims = xmin, xmax
    bins = np.linspace(xmin, xmax, 150)

    for axes, (x, y) in zip(subplots, pairs):
        data = df[[x, y]].dropna()
        r = pearsonr(data[x], data[y])[0]
        # axes.scatter(x=data[x], y=data[y], s=2, c='black', alpha=0.05, lw=0)
        sns.histplot(x=data[x], y=data[y], ax=axes, cmap='viridis', bins=(bins, bins))
        axes.plot(lims, lims, lw=0.75, linestyle='--', c='grey')
        axes.set(ylabel='Perturbed fitness', #'Fitness in {}'.format(y.upper()), 
                 xlabel='Baseline fitness', 
                 xlim=lims, ylim=lims, aspect='equal')
        axes.grid(alpha=0.2)
        axes.text(0.05, 0.95, '{} r={:.2f}'.format(y.upper(), r), transform=axes.transAxes, va='top')

    fig.tight_layout()
    fig.savefig('plots/envs_comparison.png', dpi=300)
