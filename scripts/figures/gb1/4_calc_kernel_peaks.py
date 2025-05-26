#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from os.path import exists
from epik.src.kernel import (
    AdditiveKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    ExponentialKernel,
    ConnectednessKernel,
    JengaKernel,
    GeneralProductKernel,
)
from epik.src.utils import seq_to_one_hot
from scripts.figures.settings import ALPHABET


def load_kernel(kernel, id=59):
    n_alleles, seq_length = 20, 4
    fpath = "output_new/gb1.{}.{}.model_params.pth".format(id, kernel)
    params = torch.load(fpath, map_location=torch.device("cpu"))

    kernels = {
        "Additive": AdditiveKernel,
        "Pairwise": PairwiseKernel,
        "VC": VarianceComponentKernel,
        "Exponential": ExponentialKernel,
        "Connectedness": ConnectednessKernel,
        "Jenga": JengaKernel,
        "GeneralProduct": GeneralProductKernel,
    }

    if kernel in ["Additive", "Pairwise", "VC"]:
        kernel = kernels[kernel](
            n_alleles=n_alleles,
            seq_length=seq_length,
            log_lambdas0=params["covar_module.log_lambdas"],
        )

    elif kernel in ["Exponential", "Connectedness", "Jenga", "GeneralProduct"]:
        kernel = kernels[kernel](
            n_alleles=n_alleles,
            seq_length=seq_length,
            log_var0=torch.Tensor([0.]),
            theta0=params["covar_module.theta"],
        )
    else:
        msg = "Unknwon kernel: {}".format(kernel)
        raise ValueError(msg)

    return kernel


if __name__ == "__main__":
    dataset = "gb1"
    alleles = sorted(ALPHABET[dataset])

    fpath = "datasets/{}.seqs.txt".format(dataset)
    seqs1 = [line.strip() for line in open(fpath)]
    seqs2 = ["VDGV", "WWLG", "LICA"]
    x1 = seq_to_one_hot(seqs1, alleles)
    x2 = seq_to_one_hot(seqs2, alleles)

    kernels = [
        "Additive",
        "Pairwise",
        "Exponential",
        "VC",
        "Connectedness",
        "Jenga",
        "GeneralProduct",
    ]
    labels = {"VC": "Variance Component"}

    Ks = []
    print("Computing covariances at {}".format(seqs2))
    with torch.no_grad():
        for kernel in kernels:
            label = labels.get(kernel, kernel)

            print("\t{} kernel".format(label))
            kernel = load_kernel(kernel)
            variance = kernel.forward(x2[:1, :], x2[:1, :], diag=True)

            K = kernel.forward(x1, x2).numpy() / variance
            columns = ["{}_{}".format(label, x) for x in seqs2]
            Ks.append(pd.DataFrame(K, columns=columns, index=seqs1))

    K = pd.concat(Ks, axis=1)
    K.to_csv("output_new/gb1.kernels_at_peaks.csv")
    print('Done')
