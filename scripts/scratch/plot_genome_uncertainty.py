#!/usr/bin/env python
import gzip
import numpy as np
import pandas as pd

from tqdm import tqdm
from epik.src.utils import seq_to_binary


if __name__ == '__main__':
    fpath = 'raw/genome_uncertainties.csv'
    data = pd.read_csv(fpath, index_col=0)
    print(data)
