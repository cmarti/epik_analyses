#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv('results/test_mll_simulations2.csv', index_col=0)
    data['error'] = data['logp'] - data['true_logp']
    data['rel_error'] = data['error'] / np.abs(data['true_logp']) * 100
    print(data)
    
    fig, subplots = plt.subplots(1, 2, figsize=(5.5, 2.75))
    
    axes = subplots[0]
    axes.scatter(data['n'],
                 data['error'], c='black', s=10, alpha=0.2, lw=0)
    axes.set(xlabel='Number of data points',
             ylabel='Error in MLL (log-prob)')
    axes.axvline(800, linestyle='--', lw=0.75, c='grey', label='Cholesky')
    axes.axhline(0, linestyle='--', lw=0.75, c='grey')
    
    axes = subplots[1]
    axes.scatter(data['n'],
                 data['rel_error'], c='black', s=10, alpha=0.2, lw=0)
    axes.set(xlabel='Number of data points',
             ylabel='Relative error in MLL (%)')
    axes.axvline(800, linestyle='--', lw=0.75, c='grey', label='Cholesky')
    axes.axhline(0, linestyle='--', lw=0.75, c='grey')
    
    fig.suptitle('8 positions; 4 alleles; 100 samples to estimate logdet', x=0.525, y=0.95,
                 fontsize=11, va='top', ha='center')
    fig.tight_layout()
    fig.savefig('plots/test_mll_simulations.png', dpi=300)