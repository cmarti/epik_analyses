#!/usr/bin/env python
import pandas as pd
import numpy as np

from os.path import join
from scripts.settings import (
    PARAMSDIR,
    YEAST,
    RESULTSDIR,
    DATADIR,
    PROCESSEDDIR,
)


if __name__ == "__main__":
    dataset = YEAST
    
    print("Formatting partial {} reconstruction".format(dataset))
    fpath = join(PARAMSDIR, "{}.Connectedness.2.pred.csv".format(dataset))
    landscape = pd.read_csv(fpath, index_col=0)

    fpath = join(PROCESSEDDIR, "{}.selected_loci.npy".format(dataset))
    loci = np.load(fpath, allow_pickle=True)
    loci = np.append(["BC"], loci)

    fpath = join(DATADIR, "{}.seqs_key.txt".format(dataset))
    with open(fpath) as fhand:
        seqs = [line.strip() for line in fhand]

    # Process data
    landscape.index = seqs
    ena1_idx = list(loci).index("ENA1")
    landscape["ena1"] = [seq[ena1_idx] for seq in landscape.index]
    best_ena1_rm = landscape.loc[landscape["ena1"] == "A", "coef"].idxmax()
    landscape["d"] = [
        sum(c1 != c2 for c1, c2 in zip(seq, best_ena1_rm))
        for seq in landscape.index
    ]
    landscape["d"] += np.random.normal(0, 0.1, size=landscape.shape[0])

    fpath = join(RESULTSDIR, "{}.reconstruction.csv".format(dataset))
    landscape.to_csv(fpath)
