#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from gpmap.src.space import SequenceSpace
from gpmap.src.randwalk import WMWalk


if __name__ == '__main__':
    print('Generating visualization for PREDICTED AAV2 landscape')
    landscape = pd.read_csv('output/aav.Jenga.pred_comb.csv', index_col=0)
    print(landscape)
    positions = [21, 24, 26, 27]
    X = np.array([''.join([x[i] for i in positions]) for x in landscape.index])
    y = landscape['coef'].values
    space = SequenceSpace(X=X, y=y, alphabet_type='protein')
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=2.5)
    rw.write_tables('output/aav.pred_comb', write_edges=True, nodes_format='csv')
    print(rw.decay_rates_df)
    exit()

    kernel_label = 'ARD'
    
    for dataset, kwargs in (('gb1', {'mean_function': 0.}),
                            ('smn1', {'Ns': 0.05})):
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
        
