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
    environments = [line.strip() for line in open('environments.txt')]
    # fpath = 'raw/prob_genotypes_220.csv'
    # out = 'yeast_{}'.format(environment)
    
    fpath = 'raw/prob_genotypes.csv'
    ps = pd.read_csv(fpath, index_col=0)
    genotype_uncertainty = 4 * np.mean(ps * (1 - ps), 1)
    threshold = np.percentile(genotype_uncertainty, 20)
    seqs = ps_to_seqs(ps)
    
    for environment in environments:
        print('Processing environment {}'.format(environment))
        out = 'qtls_{}'.format(environment)
        pheno = pd.read_csv('raw/pheno_data_{}.txt'.format(environment),
                            index_col=0, sep='\t')
        
        data = pd.DataFrame({'seq': seqs, 'uncertainty': genotype_uncertainty}, index=ps.index)
        data = data.join(pheno).dropna()
        data.columns = ['seq', 'uncertainty', 'y', 'y_var']
        data.set_index('seq', inplace=True)
        
        data.drop('uncertainty', axis=1).to_csv('datasets/{}.csv'.format(out))    
        data.loc[data['uncertainty'] < threshold, :].drop('uncertainty', axis=1).to_csv('datasets/{}_hq.csv'.format(out))    
