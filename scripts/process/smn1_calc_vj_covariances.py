#!/usr/bin/env python
import numpy as np
import pandas as pd

from os.path import join
from itertools import product
from gpmap.linop import calc_covariance_vjs
from scripts.settings import SMN1, DATADIR, ALPHABET, LENGTH, RESULTSDIR


if __name__ == "__main__":
    dataset = SMN1
    alphabet = ALPHABET[dataset]
    n_alleles = len(alphabet)
    seq_length = LENGTH[dataset]

    print('Loading {} data'.format(dataset))
    data = pd.read_csv(join(DATADIR, "{}.csv".format(dataset)), index_col=0)

    print('Computing empirical covariances')
    seqs = ["".join(x) for x in product(alphabet, repeat=seq_length)]
    idx = pd.Series(np.arange(len(seqs)), index=seqs)
    obs_idx = idx.loc[data.index.values].values
    y = data.y.values
    res = calc_covariance_vjs(y - y.mean(), n_alleles, seq_length, obs_idx)
    cov, ns, sites_matrix = res
    
    print('Save covariances')
    alleles = ["A", "B"]
    seqs = np.array(
        ["".join([alleles[i] for i in x]) for x in sites_matrix.astype(int)]
    )

    covs = pd.DataFrame(
        {
            "d": [x.count("A") for x in seqs],
            "data": cov / cov[0],
            "data_ns": ns,
        },
        index=seqs,
    )
    fpath = join(RESULTSDIR, "{}.vj_covariances.csv".format(dataset))
    covs.to_csv(fpath)
