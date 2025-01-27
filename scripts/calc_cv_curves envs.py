#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch

from os.path import exists
from tqdm import tqdm
from scipy.stats import pearsonr


def get_conditions(kernels, n=54):
    for kernel in kernels:
        for i in range(1, n):
            yield(kernel, i)


if __name__ == '__main__':

    datasets = ['yeast_30C', 'yeast_li', 'smn1', 'gb1',
                # 'aav'
                ]
    labels = ['Joint', 'Independent']
    
    ds_name = '37C'
    datasets = [('qtls_merged2_hq',), ('qtls_37C_hq', 'qtls_30C_hq')]
    # datasets = [('qtls_li_hq', 'qtls_30C_hq'), ('qtls_merged_hq',)]
    
    kernels = ['Rho']
    training_p = pd.read_csv('splits.csv').set_index('id')['p'].to_dict()
    n = len(training_p)

    results = []    

    for dataset, label in zip(datasets, labels):
        print('Evaluating dataset: {}'.format(dataset))

        # Iterate over subsets and kernels
        for kernel, i in tqdm(get_conditions(kernels, n=n), total=n * len(kernels)):

            # Load subset
            dfs = []
            for ds in dataset:
                data = pd.read_csv('datasets/{}.csv'.format(ds), index_col=0)
                fpath = 'output/{}.{}.{}.test_pred.csv'.format(ds, i, kernel)
                if not exists(fpath):
                    continue
                pred = pd.read_csv(fpath, index_col=0)
                pred = pred.join(data).dropna()
                dfs.append(pred)

            if dfs:
                pred = pd.concat(dfs)
                x, y = pred['y_pred'], pred['y']
                p = training_p[i]
                r2 = pearsonr(x, y)[0] ** 2
                rmse = np.sqrt(np.mean((x - y) ** 2))
                record = {'dataset': label, 'p': p,
                        'kernel': kernel, 'r2': r2, 'rmse': rmse,
                        'n': x.shape[0]}
                results.append(record)
            
    results = pd.DataFrame(results)
    print(results)
    results.to_csv('r2/{}.cv_curves.csv'.format(ds_name))
