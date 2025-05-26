#!/usr/bin/env python
import gzip
import numpy as np
import pandas as pd

from tqdm import tqdm
from epik.src.utils import seq_to_binary


if __name__ == '__main__':
    prob_gts = pd.read_csv('raw/prob_genotypes.csv', index_col=0)
    pheno = pd.read_csv('raw/pheno_data_li.txt', index_col=0, sep='\t')
    print(pheno)