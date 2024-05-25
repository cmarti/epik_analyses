#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from tqdm import tqdm
from itertools import product

if __name__ == '__main__':
    ncrhoms = 16
    annotations = pd.read_csv('datasets/yeast_annotations.csv', index_col=0)
    chr_lengths = annotations['chr'].value_counts().to_dict()
    chr_lengths = [chr_lengths['chr{}'.format(i)] for i in range(1, ncrhoms + 1)]
    print('Generating all 2 ** {} chromosome combinations'.format(ncrhoms))
    with open('datasets/yeast.seqs.txt', 'w') as fhand:
        for seq in tqdm(product(['A', 'B'], repeat=ncrhoms), total=2 ** ncrhoms):
            seq = ''.join([x * l for x, l in zip(seq, chr_lengths)])
            fhand.write('{}\n'.format(seq))

