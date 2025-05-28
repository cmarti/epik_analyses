#!/usr/bin/env python
import gzip
import numpy as np
import pandas as pd

from tqdm import tqdm

from os.path import join
from scripts.settings import DATADIR

if __name__ == '__main__':
    environment = 'li'
    
    print('Loading selected QTLs from previous study')
    fpath = join(DATADIR, 'raw', 'qtls.tsv')
    sel_loci = pd.read_csv(fpath, sep='\t')
    sel_loci = sel_loci.loc[sel_loci['Environment'] == environment, :]
    
    print('Loading SNP data')
    fpath = join(DATADIR, 'processed', 'SNP_data.csv')
    annotations = pd.read_csv(fpath, index_col=0)
    annotations = annotations.loc[annotations['SNP'] != 'END', :]
    annotations['locus'] = np.arange(annotations.shape[0])
    sel_loci = sel_loci.join(annotations, on='SNP_index')

    print('Selecting {} QTLs in {} from genome'.format(sel_loci.shape[0], environment))
    fpath = join(DATADIR, 'processed', 'prob_genotypes.csv')
    with open(fpath, 'w') as out:
        line = 'segregant,{}\n'.format(','.join([str(x) for x in range(sel_loci.shape[0])]))
        out.write(line)
        
        for i in range(1, 6):
            print('Processing file {}'.format(i))
            
            fpath = join(DATADIR, 'raw', 'geno_data_{}.txt.gz'.format(i))
            with gzip.open(fpath) as fhand:
                for line in tqdm(fhand, total=20000):
                    items = line.decode().strip().split('\t')
                    segregant = items[0]
                    ps = np.array(items[1:]).astype(float)
                    ps = ','.join([str(x) for x in ps[sel_loci['locus']]])
                    out.write('{},{}\n'.format(segregant, ps))
