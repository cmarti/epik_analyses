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
    environments = ['30C', '27C']
    labels = ['A', 'B']
    subset = 'qtls'
    out_name = 'merged2'
    
    dfs = []
    for env, label in zip(environments, labels):
        df = pd.read_csv('datasets/{}_{}_hq.csv'.format(subset, env), index_col=0)
        df.index = [x + label for x in df.index.values]
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv('datasets/{}_{}_hq.csv'.format(subset, out_name))

    for i in tqdm(range(54)):
        dfs = []
        test = []
        for env, label in zip(environments, labels):
            df = pd.read_csv('splits/{}_{}_hq.{}.train.csv'.format(subset, env, i), index_col=0)
            df.index = [x + label for x in df.index.values]
            dfs.append(df)

            test_seqs = [line.strip() + label
                         for line in open('splits/{}_{}_hq.{}.test.txt'.format(subset, env, i))]
            test.extend(test_seqs)
        df = pd.concat(dfs)
        df.to_csv('splits/{}_{}_hq.{}.train.csv'.format(subset, out_name, i))

        with open('splits/{}_{}_hq.{}.test.txt'.format(subset, out_name, i), 'w') as fhand:
            for seq in test:
                fhand.write('{}\n'.format(seq))
