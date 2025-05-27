#!/usr/bin/env python
import numpy as np
import pandas as pd

from os.path import join
from scripts.settings import YEAST, RESULTSDIR, DATADIR, LENGTH, PARAMSDIR


if __name__ == "__main__":
    dataset = YEAST
    ena1_locus = 12
    length = LENGTH[dataset]
    
    print('Loading inferred site-specific decay rates')
    fpath = join(RESULTSDIR, "{}.connectedness_decay_rates.csv".format(dataset))
    decay_rates = pd.read_csv(fpath, index_col=0)
    
    print('Load external data from QTLs')
    fpath = join(DATADIR, 'raw', 'qtls.tsv')
    sel_loci = pd.read_csv(fpath, sep="\t")
    
    fpath = join(DATADIR, 'raw', 'SNP_list.csv')
    annotations = pd.read_csv(fpath, index_col=0)
    
    fpath = join(DATADIR, 'raw', 'chr_sizes.csv')
    chr_sizes = pd.read_csv(fpath, index_col=0)
    
    print('Merging external data')
    sel_loci = sel_loci.loc[sel_loci["Environment"] == "li", :]
    sel_loci = sel_loci.join(annotations, on="SNP_index")
    sel_loci["chr"] = ["chr{}".format(i) for i in sel_loci["Chromosome"]]
    chr_x = np.cumsum(chr_sizes["size"])
    sel_loci["x"] = (
        chr_x.loc[sel_loci["chr"]].values + sel_loci["Position (bp)"]
    )
    sel_loci["decay_rate"] = decay_rates["delta"].values * 100
    sel_loci["idx"] = np.arange(sel_loci.shape[0])
    sel_loci.set_index("idx", inplace=True)

    print('Loading inferred mutational effects')
    backgrounds = [
        "A" * length,
        "B" * ena1_locus + "A"  + 'B' * (length - ena1_locus - 1),
        "A" * ena1_locus + "B"  + 'A' * (length - ena1_locus - 1),
        "B" * length,
    ]
    bcs = ["RM", "BY", "RM", "BY"]
    ena1 = ["RM", "RM", "BY", "BY"]

    data = []
    for seq, bc, ena in zip(backgrounds, bcs, ena1):
        print('\tENA1-{} in {} background'.format(ena, bc))
        fname = "{}.Connectedness.{}_expansion.csv".format(dataset, seq)
        fpath = join(PARAMSDIR, fname)
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
    
    fpath = join(RESULTSDIR, "{}_results.csv".format(dataset))
    print('Saving processed data at {}'.format(fpath))
    data.to_csv(fpath)
