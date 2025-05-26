import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_r2_curves(dataset, kernel_labels):
    config = pd.read_csv('../splits.csv'.format(dataset), index_col=0).set_index('id')
    result = pd.read_csv('../{}.r2.csv'.format(dataset))
    result['kernel'] = [x.split('.')[-1] for x in result['label']]
    result['id'] = [int(x.split('.')[0]) for x in result['label']]
    result = config.join(result.set_index('id')).reset_index()
    result['label'] = [kernel_labels.get(k, None) for k in result['kernel']]
    result.dropna(inplace=True)
    return(result)


def plot_r2_curves(axes, dataset, sel_kernels, kernel_labels, label=''):
    hue_order = [kernel_labels[k] for k in sel_kernels]
    rows = np.logical_and(np.isin(dataset['kernel'], sel_kernels),
                          dataset['r2'] > 0.1)
    sns.lineplot(x='p', y='r2', hue='label',
                 hue_order=hue_order,
                 data=dataset.loc[rows, :], ax=axes, err_style="bars")
    axes.grid(alpha=0.1)
    axes.set(xlabel='Proportion of training data',
             ylabel=r'Test $R^2$', ylim=(0.2, 1),
             title='{}'.format(label))
    axes.legend(loc=4)


if __name__ == '__main__':
    kernel_labels = {'sVC': 'regularized Skewed VC',
                     'sVC_flat': 'Skewed VC',
                     'sVCgpu_flat': 'Skewed VC (GPU)',
                     'gsp': r'$\rho,\pi$ kernel',
                     'rhos': r'$\rho$ kernel',
                     'DP': r'$\Delta^{(P)}$', 
                     'matern': 'Matern',
                     'VCgpu': 'regularized VC (GPU)',
                     'VCgpu_flat': 'VC (GPU)',
                     'VC': 'regularized VC(CPU)',
                     'VC_flat': 'VC(CPU)',
                     'rVC': r'$\lambda$ kernel',
                     'RQ': 'RQ', 
                     'RBF': 'RBF',
                     'ARD': 'ARD',
                     'HetRBF': 'HetRBF',
                     'HetARD': 'HetARD',
                     'site': 'Site Product'}
    
    smn1 = read_r2_curves('smn1', kernel_labels)

    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    sel_kernels = ['RBF', 'ARD', 'HetARD', 'HetRBF']
    plot_r2_curves(axes, smn1, sel_kernels, kernel_labels)
    fig.tight_layout()
    fig.savefig('../plots/heteroskedasticity.png', dpi=300)
