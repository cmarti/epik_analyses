import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join, exists
from scipy.stats.stats import pearsonr


def read_r2_curve(dataset):
    print('\tLoading config file')
    config = pd.read_csv('splits.csv', index_col=0).set_index('id')

    print('\tLoading R2 values for dataset {}'.format(dataset))
    result = pd.read_csv('{}.r2.csv'.format(dataset))
    result = result.loc[result['label'] != 'Rho', :]

    print('\tPreparing table for plotting')
    labels = {'mavenn': 'Additive GE',
              'mavenn_pw': 'Pairwise GE'}
    result['kernel'] = [labels.get(x.split('.')[-1], x.split('.')[-1])
                        for x in result['label']]
    result['id'] = [int(x.split('.')[-2])# if x.endswith('mavenn') else int(x.split('.')[0])
                    for x in result['label']]
    result = config.join(result.set_index('id')).reset_index()
    result['rmse'] = np.sqrt(result['mse'])
    
    # result = result.loc[result['r2'] > 0.1, :]
    #result.dropna(inplace=True)
    return(result)


def plot_r2_curves(axes, data, title='', metric='mse'):
    order = ['Additive', 'Pairwise', 'Additive GE', 'Pairwise GE', 'RBF', 'VC', 'Rho', 'ARD']
    colors = ['grey', 'cyan', 'salmon', 'darkred', 'darkorange', 'blue', 'purple', 'black']
    palette = dict(zip(order, colors))
    obs = data['kernel'].unique()
    order = [x for x in order if x in obs]
    lw = 0.75
    sns.lineplot(x='p', y=metric, hue='kernel',
                 hue_order=order, lw=lw, palette=palette,
                 data=data, ax=axes, err_style="bars",
                 err_kws={'capsize': 2, 'lw': lw}, errorbar='sd')
    axes.grid(alpha=0.2)

    ylabel = r'Test $R^2$' if metric == 'r2' else metric.upper()

    axes.set(xlabel='Proportion of training data',
             ylabel=ylabel,
             ylim=(0, 1) if metric == 'r2' else (None, None),
             xlim=(0, 1),
             title=title)
    axes.legend(loc=4, fontsize=8, frameon=False)


if __name__ == '__main__':
    dataset_labels = {'gb1': 'Protein GB1',
                      'aav': 'AAV2 Capside', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast': 'Yeast growth in Li',
                      'yeast.37C': 'Yeast growth at 37ºC'}
    # dataset = 'yeast.37C'
    metric = 'r2'
    # metric = 'loglikelihood'
    # dataset = sys.argv[1]
    # metric = sys.argv[2]

    for dataset in dataset_labels.keys():
        print('=== R2 curve for {} dataset ==='.format(dataset))
        data = read_r2_curve(dataset)
        print(data)

        try:
            print('\tPlotting R2 curves')
            fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
            plot_r2_curves(axes, data, title=dataset_labels[dataset], metric=metric)
            fig.tight_layout()
            fig.savefig('plots/{}.{}.svg'.format(dataset, metric), format='svg', dpi=300)
            fig.savefig('plots/{}.{}.png'.format(dataset, metric), format='png', dpi=300)
        except ValueError:
            continue
    print('Done')

