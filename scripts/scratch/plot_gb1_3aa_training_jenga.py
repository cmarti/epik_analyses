#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = pd.read_csv(
        "output/gb1.3aa.jenga.SLQ_lanczos_diagnosis.csv", index_col=0
    )

    fig, subplots = plt.subplots(
        1, 2, figsize=(6.5, 3), sharex=True, sharey=True
    )

    for axes, (approach, df) in zip(subplots, data.groupby("approach")):
        axes.axhline(
            df["mll"].values[-30:].mean(), c="grey", lw=0.75, linestyle="--"
        )
        axes.plot(df["n_lanczos"], df["mll"], c="black", label="Estimated")
        axes.set(
            xlabel="Number of Lanczos vectors",
            title="Optimized with {}".format(approach),
        )

    subplots[0].axvline(20, linestyle="--", lw=0.75, c="grey")
    subplots[1].axvline(100, linestyle="--", lw=0.75, c="grey")
    subplots[0].set(ylabel="MLL")

    fig.tight_layout()
    fig.savefig("plots/SLQ_diagnosis.jenga.png", dpi=300)
