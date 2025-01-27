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
    lw = 0.6
    sns.lineplot(x='p', y=metric, hue='dataset', lw=lw,
                 data=data, ax=axes, err_style="bars",
                 err_kws={'capsize': 1.5, 'capthick': lw, 'lw': lw}, errorbar='sd')
    axes.grid(alpha=0.2)

    ylabel = r'Test $R^2$' if metric == 'r2' else metric.upper()
    axes.set(xlabel='Proportion of training data',
             ylabel=ylabel,
             ylim=(0, 1) if metric == 'r2' else (None, None),
             xlim=(0, 1),
             title=title)
    axes.legend(loc=4, fontsize=8, frameon=False)


if __name__ == '__main__':
    metric = 'r2'
    ds_name = '37C'
    dataset_labels = {
                      'Independent': '1D-Fitness function',
                      'Joint': '2D-Fitness function'
                      }
    
    print('Reading CV curves from')
    fpath = 'r2/{}.cv_curves.csv'.format(ds_name)
    print('\t{}'.format(fpath))
    data = pd.read_csv(fpath, index_col=0)
    print(data)

    print('Plotting {} curves'.format(metric.upper()))
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))
    plot_r2_curves(axes, data, metric=metric)
    fig.tight_layout()
    fig.savefig('plots/{}.{}.svg'.format(ds_name, metric), format='svg', dpi=300)
    fig.savefig('plots/{}.{}.png'.format(ds_name, metric), format='png', dpi=300)
    print('Done')
