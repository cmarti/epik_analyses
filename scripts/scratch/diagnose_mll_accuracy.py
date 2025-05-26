#!/usr/bin/env python
import gpytorch
import pandas as pd
import torch
from epik.src.kernel import JengaKernel, GeneralProductKernel, ConnectednessKernel, ExponentialKernel, PairwiseKernel
from epik.src.model import EpiK
from epik.src.utils import seq_to_one_hot

AMINOACIDS = [
    "R",
    "K",
    "E",
    "D",
    "Q",
    "N",
    "H",
    "S",
    "T",
    "A",
    "V",
    "I",
    "L",
    "M",
    "P",
    "G",
    "Y",
    "F",
    "W",
    "C",
]
POSITIONS = {
    "gb1": ["39", "40", "41", "54"],
    "aav": [str(x) for x in range(561, 589)],
    "smn1": ["-3", "-2", "-1", "+2", "+3", "+4", "+5", "+6"],
}
REF_SEQS = {
    "gb1": "VDGV",
    "smn1": "CAGUAAGU",
    "aav": "DEEEIRTTNPVATEQYGSVSTNLQRGNR",
}
ALPHABET = {
    "gb1": AMINOACIDS,
    "aav": AMINOACIDS,
    "qtls_li_hq": ["A", "B"],
    "smn1": ["A", "C", "G", "U"],
}


if __name__ == "__main__":
    print("Loading data")
    # dataset, kernel_label = "qtls_li_hq", "Connectedness"  #'Jenga'
    dataset, kernel_label = "aav", "Jenga"
    i = 2
    alleles = sorted(ALPHABET[dataset])
    n_alleles = len(alleles)
    seq_length = 28 #83 # len(POSITIONS[dataset])

    data = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0)
    if "y_std" in data.columns:
        data["y_var"] = data["y_std"]  # ** 2
    seqs = data.index.values
    X = seq_to_one_hot(seqs, alleles)
    y = data["y"].values
    y_var = None
    # y_var = data["y_var"].values

    # Load trained parameters
    kernels = {'Jenga': JengaKernel, 'Connectedness': ConnectednessKernel,
               'GeneralProduct': GeneralProductKernel, 'Exponential': ExponentialKernel,
               'Pairwise': PairwiseKernel}
    print("Loading kernel parameters")
    kernel = kernels[kernel_label](seq_length=seq_length, n_alleles=n_alleles, log_var0=torch.tensor(0.0))
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)

    records = []

    # Load first optimized parameters
    # fpath = "output/{}.full.{}.{}.test_pred.csv.model_params.pth".format(
    #     dataset, i, kernel_label
    # )
    fpath = "output_new/{}.full.1.{}.model_params.pth".format(dataset, kernel_label)

    model.load(fpath, map_location="cpu")
    print("Computing MLLs")
    with torch.no_grad():
        for n in range(20, 1000, 20):
            with gpytorch.settings.max_lanczos_quadrature_iterations(n):
                mll = model.calc_mll().item()
                records.append({"n_lanczos": n, "mll": mll})
                print(records[-1])

    records = pd.DataFrame(records)
    records.to_csv(
        "output/{}.{}.mll_diagnosis.csv".format(dataset, kernel_label)
    )
