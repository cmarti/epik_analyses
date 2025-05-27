#!/usr/bin/env python
import numpy as np
import pandas as pd

from itertools import product

from os.path import join
from scripts.settings import RESULTSDIR, LENGTH, YEAST, PROCESSEDDIR, DATADIR

if __name__ == "__main__":
    dataset = YEAST
    length = LENGTH[dataset]
    loci = [
        "ENA1",
        "HAL9",
        "MKT1",
        "PHO84",
        "HAP1",
        "HAL5",
        "TAO3",
        "BUL2",
        "PTR2",
        "NRT1",
        "SUP45",
        "DPH5",
        "MLF3",
        "SUS1",
        "IRA2",
        "VIP1",
    ]

    print("Loading loci data")
    idx = np.arange(length)
    fpath = join(RESULTSDIR, "{}_results.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)
    loci_idx = np.isin(data.index, loci)

    print("\tStoring selected loci for {}".format(dataset))
    fpath = join(PROCESSEDDIR, "{}.selected_loci.npy".format(dataset))
    np.save(fpath, data.index[loci_idx].values)

    print(
        "\tStoring sequences at selected loci and keys for {}".format(dataset)
    )
    fpath = join(DATADIR, "{}.seqs.txt".format(dataset))
    with open(fpath, "w") as fhand:
        fpath = join(DATADIR, "{}.seqs_key.txt".format(dataset))

        with open(fpath, "w") as key_fhand:
            for bc in "AB":
                for subseq in product(["A", "B"], repeat=len(loci)):
                    seq = np.array([bc] * 83)
                    seq[loci_idx] = np.array(subseq)
                    seq = "".join(seq)
                    fhand.write("{}\n".format(seq))
                    key_seq = "".join((bc,) + subseq)
                    key_fhand.write("{}\n".format(key_seq))
