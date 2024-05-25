#!/usr/bin/env python
import gzip
import numpy as np
import pandas as pd

from tqdm import tqdm
from epik.src.utils import seq_to_binary


if __name__ == '__main__':
    environment = 'li'
    annot = pd.read_csv('datasets/yeast_annotations.csv', index_col=0)
    idx = annot['locus'].values
    
    sel_loci = pd.read_csv('raw/qtls.tsv', sep='\t')
    sel_loci = sel_loci.loc[sel_loci['Environment'] == environment, :]
    annotations = pd.read_csv('raw/SNP_list.txt', sep='\t', index_col=0)
    annotations = annotations.loc[annotations['SNP'] != 'END', :]
    annotations['locus'] = np.arange(annotations.shape[0])
    sel_loci = sel_loci.join(annotations, on='SNP_index')

    print('Selecting {} QTLs in {} from genome'.format(sel_loci.shape[0], environment))

    with open('raw/genome_uncertainties.csv', 'w') as out1:
        out1.write('segregant,uncertainty,sel_uncertainty,entropy,sel_entropy\n')

        with open('raw/prob_genotypes.csv', 'w') as out2:
            line = 'segregant,{}\n'.format(','.join([str(x) for x in range(sel_loci.shape[0])]))
            out2.write(line)
            
            with open('raw/prob_genotypes_220.csv', 'w') as out3:
                line = 'segregant,{}\n'.format(','.join([str(x) for x in range(idx.shape[0])]))
                out3.write(line)

                for i in range(1, 6):
                    fpath = 'raw/geno_data_{}.txt.gz'.format(i)
                    print('Processing file {}'.format(i))
                    with gzip.open(fpath) as fhand:
                        for line in tqdm(fhand, total=20000):
                            items = line.decode().strip().split('\t')
                            segregant = items[0]
                            ps = np.array(items[1:]).astype(float)
                            uncertainty = 4 * np.mean(ps * (1 - ps))
                            
                            sel_ps = ps[sel_loci['locus']]
                            sel_uncertainty = 4 * np.mean(sel_ps * (1 - sel_ps))
                            out2.write('{},{}\n'.format(segregant, ','.join([str(x) for x in sel_ps])))
                            
                            sel_ps2 = ps[idx]
                            out3.write('{},{}\n'.format(segregant, ','.join([str(x) for x in sel_ps2])))

                            ps[np.isclose(ps, 0)] += 1e-12
                            ps[np.isclose(ps, 1)] -= 1e-12
                            sel_ps[np.isclose(sel_ps, 0)] += 1e-12
                            sel_ps[np.isclose(sel_ps, 1)] -= 1e-12
                            
                            entropy = -np.mean(ps * np.log(ps) + (1-ps) * np.log(1 - ps))
                            sel_entropy = -np.mean(sel_ps * np.log(sel_ps) + (1-sel_ps) * np.log(1 - sel_ps))

                            out1.write('{},{},{},{},{}\n'.format(segregant, uncertainty, sel_uncertainty, entropy, sel_entropy))
            