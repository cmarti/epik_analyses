#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from tqdm import tqdm
from itertools import product


def ps_to_seqs(ps):
    gt = ps > 0.5
    seqs = []
    for v in gt.values:
        seqs.append(''.join(['A' if x else 'B' for x in v]))
    seqs = np.array(seqs)
    return(seqs)


if __name__ == '__main__':
    environments = ['30C', 'li']
    labels = ['A', 'B']
    subset = 'qtls'
    fpath = 'raw/prob_genotypes_220.csv'
    
    dfs = []
    for env, label in zip(environments, labels):
        df = pd.read_csv('datasets/{}_{}_hq.csv'.format(subset, env), index_col=0)
        df.index = [x + label for x in df.index.values]
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv('datasets/{}_merged_hq.csv'.format(subset))
