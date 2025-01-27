#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr

def get_additive_basis(xs):
    alleles = {'A': 1, 'B': -1}
    b = []
    for seq in xs:
        b.append([alleles[c] for c in seq])
    return(np.array(b))


def calc_basis(data):
    c = np.ones((data.shape[0], 1))
    add = get_additive_basis(data.index.values)
    idxs = np.arange(add.shape[1])
    pw = np.vstack([add[:, i] * add[:, j] for i, j in combinations(idxs, 2)]).T
    labels = ['c'] + [str(i) for i in idxs] + ['{}_{}'.format(i, j) for i,j in combinations(idxs, 2)]
    B = np.hstack([c, add, pw])
    
    n = add.shape[1] + 1
    reg = np.ones(B.shape[1])
    reg[:n] = 0

    return(B, reg, labels)


if __name__ == '__main__':
    print('Loading data')
    np.random.seed(1)
    data = pd.read_csv('datasets/qtls_li_hq.csv', index_col=0)
    u = np.random.uniform(size=data.shape[0])

    bounds = np.linspace(0, 1, 6)
    df = []
    for lower, upper in zip(bounds, bounds[1:]):
        idx = (u > lower) & (u < upper)
        train = data.loc[~idx, :]
        train_y = train['y'].values
        train_w = 1 / train['y_var'].values.reshape((train.shape[0], 1))
        train_w = train_w / train_w.shape[0]
        
        test = data.loc[idx, :]
        test_y = test['y'].values
        test_w = 1 / test['y_var'].values.reshape((test.shape[0], 1))
        test_w = test_w /test_w.shape[0]
        
        print('Building basis')
        B, reg, labels = calc_basis(train)
        B_test = calc_basis(test)[0]
        
        print('Computing A and b')
        DB = train_w * B
        A = DB.T @ B
        diag = np.diag(A)
        b = DB.T @ train_y

        for lda in np.append([0], np.geomspace(1e-2, 1e6, 19)):
            np.fill_diagonal(A, diag + lda * reg)
            x = np.linalg.solve(A, b)
            test_y_pred = B_test @ x
            train_y_pred = B @ x

            train_r2 = pearsonr(train_y_pred, train_y)[0] ** 2
            test_r2 = pearsonr(test_y_pred, test_y)[0] ** 2
            
            train_loss = np.sum(train_w * (train_y_pred - train_y) ** 2)
            test_loss = np.sum(test_w * (test_y_pred - test_y) ** 2)
            print('lambda = {} -> R2 = {:.2f} Loss = {:.2f}; '.format(lda, test_r2, test_loss))
            df.append({'lambda': lda,
                       'train_r2': train_r2, 'train_loss': train_loss,
                       'test_r2': test_r2, 'test_loss': test_loss})
            
    df = pd.DataFrame(df)
    print(df.groupby('lambda').mean())

    df = pd.DataFrame({'param': labels, 'beta': x})
    df.to_csv('results/qtls_li_hq_pw_lsq.csv')