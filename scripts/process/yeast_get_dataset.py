#!/usr/bin/env python
import numpy as np
import pandas as pd

from os.path import join
from scripts.settings import YEAST, DATADIR


def ps_to_seqs(ps):
    gt = ps > 0.5
    seqs = []
    for v in gt.values:
        seqs.append("".join(["A" if x else "B" for x in v]))
    seqs = np.array(seqs)
    return seqs


if __name__ == "__main__":
    dataset = YEAST
    environment = "li"

    print("Loading genotype probabilities")
    fpath = join(DATADIR, "processed", "prob_genotypes.csv")
    ps = pd.read_csv(fpath, index_col=0)

    print('Loading phenotypes in environment "{}"'.format(environment))
    fpath = join(DATADIR, "raw", "pheno_data_{}.txt".format(environment))
    pheno = pd.read_csv(fpath, index_col=0, sep="\t")

    print("Computing genotype uncertainties")
    genotype_uncertainty = 4 * np.mean(ps * (1 - ps), 1)
    threshold = np.percentile(genotype_uncertainty, 20)
    data = {"seq": ps_to_seqs(ps), "uncertainty": genotype_uncertainty}
    data = pd.DataFrame(data, index=ps.index).join(pheno).dropna()
    
    print("Selecting high quality genotypes")
    data = data.loc[data["uncertainty"] < threshold, :]
    data.columns = ["seq", "uncertainty", "y", "y_var"]
    data.set_index("seq", inplace=True)
    data.drop("uncertainty", axis=1, inplace=True)

    fpath = join(DATADIR, "{}.csv".format(dataset))
    print("Saving dataset to {}".format(fpath))
    data.to_csv(fpath)
