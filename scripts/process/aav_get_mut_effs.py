#!/usr/bin/env python
import pandas as pd

from os.path import join
from scripts.settings import AAV, PARAMSDIR, RESULTSDIR, POSITIONS, AAV_BACKGROUNDS


if __name__ == '__main__':
    dataset = AAV
    positions = POSITIONS[dataset]
    
    print('Merging estimated mutational effects in a single dataframe')
    dfs = []
    for label, seq in AAV_BACKGROUNDS.items():
        fname = '{}.Jenga.{}_expansion.csv'.format(dataset, seq)
        fpath = join(PARAMSDIR, fname)
        df = pd.read_csv(fpath, index_col=0)
        df.columns = ['{}_{}'.format(col, label) for col in df.columns]
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    
    df['allele1'] = [x[0] for x in df.index]
    df['allele2'] = [x[-1] for x in df.index]
    df['position'] = [positions[int(x[1:-1])] for x in df.index]
    df.index = ['{}{}{}'.format(a1, p, a2)
                for a1, p, a2 in zip(df['allele1'], df['position'], df['allele2'])]
    
    fpath = join(RESULTSDIR, '{}.mutational_effects.csv'.format(dataset))
    print('Mutational effects saved at {}'.format(fpath))
    df.to_csv(fpath)