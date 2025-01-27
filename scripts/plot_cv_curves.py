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
    labels = {'r2': 'Test $R^2$', 'rmse': 'rmse', 'logit_r2': r'$\log_2\left(\frac{V_{model}}{V_{res}}\right)$'}
    order = ['Additive', 'Pairwise', 'Global epistasis', 'Exponential', 'Variance Component', 'Connectedness', 'Jenga', 'AddRho']
    colors = ['silver', 'gray', 'salmon', 'violet', 'slateblue', 'purple', 'black', 'gold']
    palette = dict(zip(order, colors))
    obs = data['kernel'].unique()
    order = [x for x in order if x in obs]
    lw = 0.6
    sns.lineplot(x='p', y=metric, hue='kernel',
                 hue_order=order, lw=lw, palette=palette,
                 data=data, ax=axes, err_style="bars",
                 err_kws={'capsize': 1.5, 'capthick': lw, 'lw': lw}, errorbar='sd')
    axes.grid(alpha=0.2)


    ylabel = labels[metric]

    axes.legend(loc=4, fontsize=8, frameon=False, ncol=2)
    axes.set(xlabel='Proportion of training data',
             ylabel=ylabel,
            #  ylim=(0, 1) if metric == 'r2' else (None, None),
            #  xlim=(0, 1),
             title=title,
             xscale='log',
            #  yscale='logit',
             )


if __name__ == '__main__':
    metric = 'r2'
    dataset_labels = {'gb1': 'Protein GB1',
                      'aav': 'AAV2 Capside', 
                      'smn1': 'SMN1 5´splice site',
                      'yeast_li': 'Yeast growth in Li',
                      'yeast_li_hq': 'Yeast growth in Li',
                      'qtls_li': 'Yeast growth in Li',
                      'qtls_li_hq': 'Yeast growth in Li',
                      'yeast_37C': 'Yeast growth at 37ºC',
                      'yeast_30C': 'Yeast growth at 30ºC'}
    
    
    datasets = ['qtls_li_hq', 'aav', 'smn1', 'gb1'] # 'yeast_li_hq', 'qtls_li', 'qtls_li_hq']
    # datasets = ['qtls_li_hq', 'yeast_li_hq', 'qtls_li', 'yeast_li']
    # datasets = ['aav']

    for dataset in datasets:
        fpath = 'r2/{}.cv_curves.csv'.format(dataset)
        print('Reading CV curves from {}'.format(fpath))
        data = pd.read_csv(fpath, index_col=0)
        print(data)
        # data = data.loc[data['kernel'] != 'pairwise', :]
        data['logit_r2'] = np.log2(data['r2'] /(1 - data['r2']))
        data.loc[data['kernel'] == 'mavenn', 'kernel'] = 'Global epistasis' 
        data.loc[data['kernel'] == 'RBF', 'kernel'] = 'Exponential' 
        data.loc[data['kernel'] == 'Rho', 'kernel'] = 'Connectedness' 
        data.loc[data['kernel'] == 'ARD', 'kernel'] = 'Jenga' 
        data.loc[data['kernel'] == 'VC', 'kernel'] = 'Variance Component' 

        print('\tPlotting {} curve for dataset: {}'.format(metric.upper(), dataset))
        fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
        plot_r2_curves(axes, data, title=dataset_labels[dataset], metric=metric)
        fig.tight_layout()
        fig.savefig('plots/{}.{}.svg'.format(dataset, metric), format='svg', dpi=300)
        fig.savefig('plots/{}.{}.png'.format(dataset, metric), format='png', dpi=300)
    print('Done')
