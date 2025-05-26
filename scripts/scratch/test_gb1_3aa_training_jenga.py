#!/usr/bin/env python
import gpytorch
import pandas as pd
import torch
from epik.src.kernel import JengaKernel
from epik.src.model import EpiK
from epik.src.utils import seq_to_one_hot
from torch.optim import Adam

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

    records = []

    # Default parameters
    kernel = JengaKernel(seq_length=seq_length, n_alleles=n_alleles)
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)
    model.optimizer = Adam(model.gp.parameters(), lr=0.1)
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
    model.save("output/gb1.3aa.jenga.SLQ.model_params.pth")

    # Increased lanczos iterations
    kernel = JengaKernel(seq_length=seq_length, n_alleles=n_alleles)
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
    model.save("output/gb1.3aa.jenga.SLQ_100.model_params.pth")

    # With Cholesky decomposition
    kernel = JengaKernel(seq_length=seq_length, n_alleles=n_alleles)
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
    model.save("output/gb1.3aa.jenga.cholesky.model_params.pth")

    results = pd.DataFrame(records)
    results.to_csv("output/mll_training_test.gb13aa.jenga.csv")
