import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr


def read_r2(dataset):
    result = pd.read_csv('{}.r2.csv'.format(dataset), index_col=0)
    labels = {'mavenn': 'Additive GE', 'mavenn_pw': 'Pairwise GE'}
    result['kernel'] = [labels.get(x.split('.')[-1], x.split('.')[-1])
                        for x in result['label']]
    result['rmse'] = np.sqrt(result['mse'])
    return(result)


def plot_r2(axes, data, metric='mse'):
    order = ['Additive', 'Additive GE', 'Pairwise GE', 'RBF', 'VC', 'Rho', 'ARD']
    colors = ['grey', 'salmon', 'darkred', 'cyan', 'blue', 'purple', 'black']
    palette = dict(zip(order, colors))
    obs = data['kernel'].unique()
    order = [x for x in order if x in obs]
    sns.stripplot(x='env', y=metric, hue='kernel', size=4, 
                  dodge=True, alpha=0.5, jitter=.3,
                  hue_order=order, palette=palette,
                  data=data, ax=axes)
    axes.grid(alpha=0.2)

    ylabel = r'Test $R^2$' if metric == 'r2' else metric.upper()

    axes.set(xlabel='Proportion of training data',
             ylabel=ylabel,
             ylim=(0, 0.8) if metric == 'r2' else (None, None))
    axes.legend(loc=4, fontsize=8, frameon=False)


if __name__ == '__main__':
    environments = [line.strip() for line in open('environments.txt')]
    metric = 'r2'

    merged = []
    for env in environments:
        dataset = 'yeast.{}'.format(env)
        data = read_r2(dataset)
        data['env'] = env.upper()
        merged.append(data)
    merged = pd.concat(merged)
    merged = merged.loc[merged['r2'] > 0.05, :]

    fig, axes = plt.subplots(1, 1, figsize=(8.5, 3.5), sharex=True)
    plot_r2(axes, merged, metric='r2')
    fig.tight_layout()
    fig.savefig('plots/yeast_envs.svg'.format(dataset), format='svg', dpi=300)
    fig.savefig('plots/yeast_envs.png'.format(dataset), format='png', dpi=300)

