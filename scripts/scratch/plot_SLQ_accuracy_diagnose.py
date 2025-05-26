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

    # Load trained parameters
    kernel = JengaKernel(seq_length=seq_length, n_alleles=n_alleles)
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)

    records = []

    # Load first optimized parameters
    fpath = "output/gb1.3aa.jenga.SLQ.model_params.pth"
    model.load(fpath)
    with torch.no_grad():
        for n in range(5, 200):
            with gpytorch.settings.max_lanczos_quadrature_iterations(n):
                mll = model.calc_mll().item()
                records.append({"n_lanczos": n, "mll": mll, "approach": "SLQ"})
                print(records[-1])

    # Load optimized parameters with 100 vectors
    fpath = "output/gb1.3aa.jenga.SLQ_100.model_params.pth"
    model.load(fpath)
    with torch.no_grad():
        for n in range(5, 200):
            with gpytorch.settings.max_lanczos_quadrature_iterations(n):
                mll = model.calc_mll().item()
                records.append(
                    {"n_lanczos": n, "mll": mll, "approach": "SLQ-100"}
                )
                print(records[-1])

    records = pd.DataFrame(records)
    records.to_csv("output/gb1.3aa.jenga.SLQ_lanczos_diagnosis.csv")
