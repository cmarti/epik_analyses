#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns

from os.path import exists
from epik.src.utils import seq_to_binary


if __name__ == '__main__':
    annotations = pd.read_csv('datasets/yeast_annotations.csv', index_col=0)
    annotations['pos'] = np.arange(annotations.shape[0])
    
    environments = [line.strip() for line in open('environments.txt')]
    
    betas = []
    for environment in environments:
        print(environment)
        data = pd.read_csv('datasets/yeast_{}.csv'.format(environment), index_col=0)
        X = seq_to_binary(data.index.values, ref='A')
        ones = torch.ones((X.shape[0], 1))
        X = torch.tensor(np.hstack((ones, X)))
        y = torch.tensor(data['y'].values, dtype=X.dtype)
        betas.append(torch.linalg.lstsq(X, y).solution.numpy())
        
    betas = pd.DataFrame(betas, index=environments)
    annotations = annotations.join(betas.T)
    annotations.to_csv('loci_betas_annotations.csv')
