#!/usr/bin/env python
import numpy as np
import pandas as pd

from scripts.figures.utils import load_decay_rates

if __name__ == "__main__":
    dataset = "qtls_li_hq"
    decay_rates = load_decay_rates(dataset, kernel="Connectedness")
    sel_loci = pd.read_csv("raw/qtls.tsv", sep="\t")
    sel_loci = sel_loci.loc[sel_loci["Environment"] == "li", :]
    annotations = pd.read_csv("raw/SNP_list.csv", index_col=0)
    sel_loci = sel_loci.join(annotations, on="SNP_index")
    sel_loci["chr"] = ["chr{}".format(i) for i in sel_loci["Chromosome"]]
    chr_sizes = pd.read_csv("raw/chr_sizes.csv", index_col=0)["size"]
    chr_x = np.cumsum(chr_sizes)
    sel_loci["x"] = (
        chr_x.loc[sel_loci["chr"]].values + sel_loci["Position (bp)"]
    )
    sel_loci["decay_rate"] = decay_rates["delta"].values * 100
    sel_loci["idx"] = np.arange(sel_loci.shape[0])
    sel_loci.set_index("idx", inplace=True)

    # Load mutational effects
    backgrounds = [
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "BBBBBBBBBBBBABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
        "AAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    ]
    bcs = ["RM", "BY", "RM", "BY"]
    ena1 = ["RM", "RM", "BY", "BY"]

    data = []
    for seq, bc, ena in zip(backgrounds, bcs, ena1):
        fpath = "output_new/qtls_li_hq.Connectedness.{}_expansion.csv".format(
            seq
        )
        df = pd.read_csv(fpath, index_col=0)
        df = df.loc[["_" not in x for x in df.index], :]
        idx = np.array([x.startswith("B") for x in df.index])
        df.index = [
            x[-1] + x[1:-1] + x[0] if x.startswith("B") else x for x in df.index
        ]
        df.loc[idx, "coef"] = -df.loc[idx, "coef"]
        df.loc[idx, "lower_ci"], df.loc[idx, "upper_ci"] = (
            -df.loc[idx, "upper_ci"],
            -df.loc[idx, "lower_ci"],
        )
        df.columns = ["{}_{}_ena1{}".format(c, bc, ena) for c in df.columns]
        data.append(df)
    data = pd.concat(data, axis=1)
    data["idx"] = [int(x[1:-1]) for x in data.index]
    data = data.join(sel_loci, on="idx", rsuffix="_locus").set_index("gene")
    data.to_csv("results/{}_results.csv".format(dataset))
