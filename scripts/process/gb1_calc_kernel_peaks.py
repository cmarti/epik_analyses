#!/usr/bin/env python
import torch
import pandas as pd

from itertools import product
from os.path import join

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

from scripts.utils import torch_load
from scripts.settings import (
    ALPHABET,
    GB1,
    LENGTH,
    MODELS,
    MODEL_KEYS,
    PARAMSDIR,
    RESULTSDIR,
)


def load_kernel(kernel, n_alleles, seq_length, id=59):
    fpath = join(PARAMSDIR, "gb1.{}.{}.model_params.pth".format(id, kernel))
    params = torch_load(fpath)

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
            log_var0=torch.Tensor([0.0]),
            theta0=params["covar_module.theta"],
        )
    else:
        msg = "Unknwon kernel: {}".format(kernel)
        raise ValueError(msg)

    return kernel


if __name__ == "__main__":
    dataset = GB1
    alleles = sorted(ALPHABET[dataset])
    length = LENGTH[dataset]

    seqs1 = ["".join(x) for x in product(alleles, repeat=length)]
    seqs2 = ["VDGV", "WWLG", "LICA"]
    x1 = seq_to_one_hot(seqs1, alleles)
    x2 = seq_to_one_hot(seqs2, alleles)

    kernels = [x for x in MODELS if x != "Global epistasis"]

    Ks = []
    print("Computing covariances with {}".format(seqs2))
    with torch.no_grad():
        for kernel in kernels:
            label = MODEL_KEYS.get(kernel, kernel)

            print("\t{} kernel".format(kernel))
            kernel = load_kernel(
                label, n_alleles=len(alleles), seq_length=length
            )
            variance = kernel.forward(x2[:1, :], x2[:1, :], diag=True)

            K = kernel.forward(x1, x2).numpy() / variance
            columns = ["{}_{}".format(label, x) for x in seqs2]
            Ks.append(pd.DataFrame(K, columns=columns, index=seqs1))

    K = pd.concat(Ks, axis=1)

    K.to_csv(join(RESULTSDIR, "gb1.kernels_at_peaks.csv"))
    print("Done")
