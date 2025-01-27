#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from tqdm import tqdm
from itertools import product

if __name__ == '__main__':
    background = 'DEEEIRTTNPVATEQYGSVSTNLQRGNR'
    positions = [21, 24, 26, 27]
    
    background = 'DEEEIRTTNPVATEQYGSVST{}LQ{}G{}{}'
    alleles_order = ['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                     'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C']
    seqs = [background.format(*cs) for cs in product(alleles_order, repeat=4)]
    with open('datasets/aav.test_comb.txt', 'w') as fhand:
        for s in tqdm(seqs):
            fhand.write('{}\n'.format(s))

    data = pd.read_csv('datasets/aav.csv', index_col=0)
    data['seq'] = [''.join([x[p] for p in positions]) for x in data.index.values]
    df = data.loc[np.intersect1d(data.index.values, seqs), :]
    print(data['seq'].unique().shape)
    print(df)
    print(data.min(), data.max())
    print(df.min(), df.max())