#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from torch.optim import Adam

from epik.src.kernel import (
    AdditiveKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    ExponentialKernel,
    ConnectednessKernel,
    JengaKernel,
    GeneralProductKernel,
)
import gpytorch
from epik.src.utils import seq_to_one_hot, encode_seqs
from epik.src.model import EpiK
from gpmap.src.linop import KronOperator, MultivariateGaussian, KernelOperator


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


if __name__ == "__main__":
    dataset = "gb1"
    i = 1
    alleles = sorted(AMINOACIDS)
    n_alleles = len(alleles)
    seq_length = 3

    data = pd.read_csv("datasets/gb1.csv", index_col=0)
    data["x"] = [x[0] + x[2:] for x in data.index]
    data = data.groupby("x").mean()
    seqs = data.index.values
    X = seq_to_one_hot(seqs, alleles)
    y = data["y"].values
    y_var = data["y_var"].values

    # fpath = "output/gb1.1.Connectedness.test_pred.csv.model_params.pth"
    # params = torch.load(fpath, map_location=torch.device("cpu"))
    # idx = [0, 2, 3]
    # kernel = ConnectednessKernel(
    #     seq_length=seq_length,
    #     n_alleles=n_alleles,
    #     log_var0=params["covar_module.module.log_var"],
    #     theta0=params["covar_module.module.theta"][idx],
    # )

    records = []

    # Default parameters
    kernel = ConnectednessKernel(seq_length=seq_length, n_alleles=n_alleles)
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)
    model.optimizer = Adam(model.gp.parameters(), lr=0.05)
    model.set_training_mode()
    for i in range(100):
        with torch.no_grad():
            with gpytorch.settings.max_cholesky_size(10000):
                logp = model.calc_mll().item()
        model.training_step()
        records.append(
            {
                "iteration": i,
                "mll": model.mll,
                "real_mll": logp,
                "approach": "SLQ",
            }
        )
        print(records[-1])

    results = pd.DataFrame(records)
    results.to_csv("output/mll_training_test.gb13aa.csv")

    # Increased lanczos iterations
    kernel = ConnectednessKernel(seq_length=seq_length, n_alleles=n_alleles)
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)
    model.optimizer = Adam(model.gp.parameters(), lr=0.05)
    model.set_training_mode()
    with gpytorch.settings.max_lanczos_quadrature_iterations(100):
        for i in range(100):
            with torch.no_grad():
                with gpytorch.settings.max_cholesky_size(10000):
                    logp = model.calc_mll().item()
            model.training_step()
            records.append(
                {
                    "iteration": i,
                    "mll": model.mll,
                    "real_mll": logp,
                    "approach": "SLQ-100",
                }
            )
            print(records[-1])

    results = pd.DataFrame(records)
    results.to_csv("output/mll_training_test.gb13aa.csv")

    # With Cholesky decomposition
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)
    model.optimizer = Adam(model.gp.parameters(), lr=0.05)
    model.set_training_mode()

    with gpytorch.settings.max_cholesky_size(10000):
        for i in range(100):
            with torch.no_grad():
                logp = model.calc_mll().item()
            model.training_step()
            records.append(
                {
                    "iteration": i,
                    "mll": model.mll,
                    "real_mll": logp,
                    "approach": "cholesky",
                }
            )
            print(records[-1])

    results = pd.DataFrame(records)
    results.to_csv("output/mll_training_test.gb13aa.csv")
