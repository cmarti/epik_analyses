#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data1 = pd.read_csv(
        "output/qtls_li_hq.Exponential.mll_diagnosis.csv", index_col=0
    )
    data1["dataset"] = "Exponential"

    data2 = pd.read_csv(
        "output/qtls_li_hq.Connectedness.mll_diagnosis.csv", index_col=0
    )
    data2["dataset"] = "Connectedness"

    fig, subplots = plt.subplots(
        1, 2, figsize=(7, 3), sharex=True, sharey=False
    )
    data = pd.concat([data1, data2])

    for axes, (dataset, df) in zip(subplots, data.groupby("dataset")):
        axes.axhline(
            df["mll"].values[-5:].mean(), c="grey", lw=0.75, linestyle="--"
        )
        axes.plot(df["n_lanczos"], df["mll"], c="black")
        axes.set(
            xlabel="Number of Lanczos vectors",
            title=dataset,
        )

    subplots[0].axvline(400, linestyle="--", lw=0.75, c="grey")
    subplots[1].axvline(400, linestyle="--", lw=0.75, c="grey")
    subplots[0].set(ylabel="MLL")

    fig.tight_layout()
    fig.savefig("plots/mll_diagnosis.qtls.png", dpi=300)
