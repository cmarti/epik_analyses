#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from scipy.stats import pearsonr


def read_decay_factors(dataset, id=None, kernel='Rho'):
    if id is None:
        fpath = 'output/{}.{}.test_pred.csv.model_params.pth'.format(dataset, kernel)
    else:
        fpath = 'output/{}.{}.{}.test_pred.csv.model_params.pth'.format(dataset, id, kernel)

    params = torch.load(fpath, map_location=torch.device('cpu'))
    logit_rho = params['covar_module.logit_rho'].numpy()
    log_p = params['covar_module.log_p'].numpy()
    
    rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
    p = np.exp(log_p)
    p = p / np.expand_dims(p.sum(1), 1)
    eta = (1 - p) / p
    
    decay_factor = 1 - (1 - rho) / (1 + eta * rho)

    if kernel == 'Rho':
        decay_factor = decay_factor.mean(1)
    elif kernel == 'ARD':
        decay_factor = decay_factor.flatten()
    else:
        raise ValueError('Unknown kernel: {}'.format(kernel))

    return(decay_factor * 100)


if __name__ == '__main__':
    training_p = pd.read_csv('splits.csv').set_index('id')['p'].to_dict()

    results = []
    for dataset, kernel in [
                            ('qtls_li', 'Rho'),
                            # ('smn1', 'ARD'),
                            # ('gb1', 'ARD'),
                            # ('aav', 'ARD'),
                            # ('yeast_30C', 'Rho'),
                            # ('yeast_li', 'Rho'),
                            # ('yeast_37C', 'Rho'),
                            ]:
        
        y = read_decay_factors(dataset, kernel=kernel)

        decay_factors = []
        for i in range(len(training_p)):
            try:
                x = read_decay_factors(dataset, id=str(i), kernel=kernel)
            except FileNotFoundError:
                continue

            # Compute metrics and store
            p = training_p[i]
            r2 = pearsonr(x, y)[0] ** 2
            rmse = np.sqrt(np.mean((x - y) ** 2))
            record = {'dataset': dataset, 'p': p,
                      'kernel': kernel, 'r2': r2, 'rmse': rmse,
                      'n': x.shape[0]}
            results.append(record)
    results = pd.DataFrame(results)
    results.to_csv('cv_curves_decay_rates.csv')
