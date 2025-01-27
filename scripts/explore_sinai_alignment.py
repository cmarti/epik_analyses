#!/usr/bin/env python
import numpy as np
import pandas as pd


if __name__ == '__main__':
    ref = 561
    data = pd.read_csv('datasets/processed_alignment_data.csv')
    data['576'] = [x[576-ref] for x in data['seq']]
    print(data['576'].value_counts())
    print(data.loc[data['576']== 'Y', :])
    print(data.loc[data['576']== 'G', :])
    print(data.loc[~np.isin(data['576'], ['Y', 'F', 'W'])])
    
    # for i, seq in enumerate(data['sequence'][:100].unique()):
    #     print('>seq{}'.format(i))
    #     print(seq)