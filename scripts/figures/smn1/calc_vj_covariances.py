#!/usr/bin/env python
import numpy as np
import pandas as pd
from gpmap.src.linop import calc_covariance_vjs
from gpmap.src.seq import generate_possible_sequences, guess_space_configuration


if __name__ == "__main__":
    dataset = "smn1"

    # Load data
    fpath = "datasets/{}.csv".format(dataset)
    data = pd.read_csv(fpath, index_col=0)
    config = guess_space_configuration(data.index.values, ensure_full_space=False)
    alphabet = np.unique(np.hstack(config["alphabet"]))

    X = np.array(
        list(generate_possible_sequences(l=config["length"], alphabet=alphabet))
    )
    idx = pd.Series(np.arange(X.shape[0]), index=X)
    obs_idx = idx.loc[data.index.values].values
    y = data.y.values
    cov, ns, sites_matrix = calc_covariance_vjs(
        y - y.mean(), config["n_alleles"][0], config["length"], obs_idx
    )
    alleles = ["A", "B"]
    seqs = np.array(
        ["".join([alleles[i] for i in x]) for x in sites_matrix.astype(int)]
    )

    covs = pd.DataFrame(
        {"d": [x.count("B") for x in seqs], "data": cov / cov[0], "data_ns": ns},
        index=seqs,
    )
    covs.to_csv("{}.vj_covariances.csv".format(dataset))
