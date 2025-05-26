#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from tqdm import tqdm
from itertools import product

if __name__ == '__main__':
    data = pd.read_csv('raw/41587_2020_793_MOESM3_ESM.csv')
    print(data['partition'].value_counts())
    print(data.loc[data['partition'] == 'stop', :])
    print(data.loc[data['partition'] == 'stop', 'viral_selection'].std())

    comb = [line.strip() for line in open('datasets/aav.test_comb.txt')]
    seqs = np.intersect1d([x.upper() for x in data['sequence']], comb)
    print(seqs.shape)