#!/usr/bin/env python
import numpy as np
import pandas as pd

from itertools import product

if __name__ == "__main__":
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
    idx = np.arange(83)
    data = pd.read_csv("results/qtls_li_hq_results.csv", index_col=0)
    loci_idx = np.isin(data.index, loci)
    # print(data.loc[loci_idx, :])
    # exit()

    np.save(
        "datasets/qtls_li_hq.selected_loci.npy", data.index[loci_idx].values
    )
    print(data.index[loci_idx].values)
    exit()

    with open("datasets/qtls_li_hq.seqs.txt", "w") as fhand:
        with open("datasets/qtls_li_hq.seqs_key.txt", "w") as key_fhand:
            for bc in "AB":
                for subseq in product(["A", "B"], repeat=len(loci)):
                    seq = np.array([bc] * 83)
                    seq[loci_idx] = np.array(subseq)
                    seq = "".join(seq)
                    fhand.write("{}\n".format(seq))
                    key_seq = "".join((bc,) + subseq)

                    key_fhand.write("{}\n".format(key_seq))
