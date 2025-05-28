#!/usr/bin/env python
import numpy as np
import pandas as pd

from tqdm import tqdm
from pysam import TabixFile

from os.path import join
from scripts.settings import DATADIR


if __name__ == '__main__':
    print('Extracting closest gene within 10Kb of every SNP')
    window = 10000

    print('Loading annotation Tabix index')
    index = TabixFile(join(DATADIR, 'raw', 'annotations.gff.gz'))
    
    print('Loading SNP data')
    fpath = join(DATADIR, 'raw', 'SNP_list.txt')
    annotations = pd.read_csv(fpath, sep='\t', index_col=0)
    annotations = annotations.loc[annotations['SNP'] != 'END', :]
    annotations['locus'] = np.arange(annotations.shape[0])
    chroms = annotations['Chromosome']
    positions = annotations['Position (bp)']
    n = annotations.shape[0]

    genes = []
    ds = []
    print('Extracting SNP annotations')
    for chrom, pos in tqdm(zip(chroms, positions), total=n):
        chrom = 'chr{:02d}'.format(chrom)
        start = max(0, pos - window)
        end = pos + window

        min_d = np.inf
        gene = None
        for record in index.fetch(chrom, start, end):
            items = record.split('\t')
            if items[2] != 'gene':
                continue

            start, end = int(items[3]), int(items[4])
            d = min(abs(start - pos), abs(end - pos))
            attrs = items[-1].split(';')
            attrs = {x.split('=')[0]: x.split('=')[-1] for x in attrs}
            if d < min_d:
                gene = attrs.get('gene', None)
                min_d = d
                
        genes.append(gene)
        ds.append(min_d)
    
    annotations['gene'] = genes
    annotations['distance'] = ds
    
    fpath = join(DATADIR, 'processed', 'SNP_data.csv')
    print('Saving SNP annotations at to {}'.format(fpath))
    annotations.to_csv(fpath)
