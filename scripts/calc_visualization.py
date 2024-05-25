#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from gpmap.src.space import SequenceSpace
from gpmap.src.randwalk import WMWalk


if __name__ == '__main__':
    kernel_label = 'ARD'
    
    for dataset, kwargs in (('smn1', {'Ns': 0.05}),
                             ('gb1', {'mean_function': 0.})):
        print('Generating visualization for inferred landscape {} with {} kernel'.format(dataset, kernel_label))
        landscape = pd.read_csv('output/{}.{}.test_pred.csv'.format(dataset, kernel_label), index_col=0)
        if dataset == 'smn1':
            seqs = [x for x in landscape.index if x[3] in 'CU']
            landscape = landscape.loc[seqs, :]
        space = SequenceSpace(X=landscape.index.values, 
                              y=landscape['y_pred'].values)
        rw = WMWalk(space)
        rw.calc_visualization(**kwargs)
        rw.write_tables('output/{}.{}'.format(dataset, kernel_label),
                        write_edges=True, nodes_format='csv')
        print(rw.decay_rates_df)
        exit()
        
