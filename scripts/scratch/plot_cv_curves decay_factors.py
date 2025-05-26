import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join, exists
from scipy.stats.stats import pearsonr


def plot_r2_curves(axes, data, title='', metric='mse'):
    order = ['Additive', 'Pairwise', 'Threeway', 'Global epistasis', 'Exponential', 'VC', 'Rho', 'ARD']
    colors = ['silver', 'gray', 'dimgrey', 'salmon', 'violet', 'slateblue', 'purple', 'black']
    palette = dict(zip(order, colors))
    obs = data['kernel'].unique()
    order = [x for x in order if x in obs]
    lw = 1
    sns.lineplot(x='p', y=metric, hue='kernel',
                 hue_order=order, lw=lw, palette=palette,
                 data=data, ax=axes, err_style="bars",
                 err_kws={'capsize': 1.5, 'capthick': lw, 'lw': lw}, errorbar='sd',
                 legend=False)
    axes.grid(alpha=0.2)

    ylabel = 'Decay rates full dataset '
    ylabel += '$R^2$' if metric == 'r2' else metric.upper()

    axes.set(xlabel='Proportion of training data',
             ylabel=ylabel,
             ylim=(0, 1) if metric == 'r2' else (None, None),
             xlim=(0, 1),
             title=title)
    # axes.legend(loc=4, fontsize=8, frameon=False)


if __name__ == '__main__':
    fpath = 'cv_curves_decay_rates.csv'
    metric = 'r2'
    dataset_labels = {'gb1': 'Protein GB1',
                      'aav': 'AAV2 Capside', 
                      'smn1': 'SMN1 5´splice site',
                      'hq_li': 'Yeast growth in Li',
                      'qtls_li': 'Yeast growth in Li',
                      'yeast_li': 'Yeast growth in Li',
                      'yeast_37C': 'Yeast growth at 37ºCtraining_p)',
                      'yeast_30C': 'Yeast growth at 30ºC'}
    
    print('Reading CV curves from {}'.format(fpath))
    data = pd.read_csv(fpath, index_col=0)
    data.loc[data['kernel'] == 'mavenn', 'kernel'] = 'Global epistasis' 
    data.loc[data['kernel'] == 'RBF', 'kernel'] = 'Exponential' 

    for dataset, df in data.groupby('dataset'):
        print('\tPlotting {} curve for dataset: {}'.format(metric.upper(), dataset))
        fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
        plot_r2_curves(axes, df, title=dataset_labels[dataset], metric=metric)
        fig.tight_layout()
        fig.savefig('plots/{}.decay_rates.{}.svg'.format(dataset, metric), format='svg', dpi=300)
        fig.savefig('plots/{}.decay_rates.{}.png'.format(dataset, metric), format='png', dpi=300)
    print('Done')
