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
    
    freqs = []
    for environment in environments:
        print(environment)
        data = pd.read_csv('datasets/yeast_{}.csv'.format(environment), index_col=0)
        X = seq_to_binary(data.index.values, ref='A').numpy()
        freqs.append(np.mean((1 + X) / 2, axis=0))
    
    freqs = pd.DataFrame(freqs, index=environments)
    freqs.to_csv('datasets/yeast.allele_freqs.csv')
        