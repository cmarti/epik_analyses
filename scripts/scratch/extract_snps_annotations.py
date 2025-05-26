#!/usr/bin/env python
import numpy as np
import pandas as pd

from tqdm import tqdm
from pysam import TabixFile


if __name__ == '__main__':
    print('Extracting closest gene within 10Kb of every SNP')
    window = 10000

    annotations = pd.read_csv('raw/SNP_list.txt', sep='\t', index_col=0)
    annotations = annotations.loc[annotations['SNP'] != 'END', :]
    annotations['locus'] = np.arange(annotations.shape[0])

    index = TabixFile('raw/annotations.gff.gz')
    
    genes = []
    ds = []
    for chrom, pos in tqdm(zip(annotations['Chromosome'], annotations['Position (bp)']),
                           total=annotations.shape[0]):
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
    annotations.to_csv('raw/SNP_list.csv')
