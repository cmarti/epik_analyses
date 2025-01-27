#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    ref = 561
    charge = {"E": -1, "D": -1, "R": 1, "K": 1}

    data = pd.read_csv("datasets/aav.csv", index_col=0)
    lims = data["y"].min(), data["y"].max()
    data["576"] = [x[576 - ref] for x in data.index]
    data["576_other"] = [
        x[: 576 - ref] + x[576 - ref + 1 :] for x in data.index
    ]
    data["569"] = [x[569 - ref] for x in data.index]
    data["569_other"] = [
        x[: 569 - ref] + x[569 - ref + 1 :] for x in data.index
    ]
    data["charge"] = [
        np.sum([charge.get(aa, 0) for aa in seq[-8:]])
        for seq in data.index.values
    ]

    bins = np.linspace(lims[0], lims[1], 50)
    fig, subplots = plt.subplots(3, 1,
                                 figsize=(0.33 * FIG_WIDTH,
                                          0.85 * FIG_WIDTH))

    axes = subplots[0]
    xticks = np.array([-10, -5, 0, 5, 10])
    axes.hist(
        data.loc[data["569"] != "N", "y"],
        color="grey",
        label="Other",
        alpha=0.5,
        bins=bins,
        density=True,
    )
    axes.hist(
        data.loc[data["569"] == "N", "y"],
        color="black",
        label="569N",
        alpha=0.5,
        bins=bins,
        density=True,
    )
    axes.legend(loc=0, fontsize=9)
    axes.set(xlabel="DMS score", ylabel="Probability density", xticks=xticks)

    axes = subplots[1]
    idx = np.isin(data["576"], ["Y", "F", "W"])
    axes.hist(
        data.loc[~idx, "y"],
        color="grey",
        label="Other",
        alpha=0.5,
        bins=bins,
        density=True,
    )
    axes.hist(
        data.loc[idx, "y"],
        color="black",
        label="576Y/F/W",
        alpha=0.5,
        bins=bins,
        density=True,
    )
    axes.legend(loc=0, fontsize=9)
    axes.set(xlabel="DMS score", ylabel="Probability density", xticks=xticks)

    axes = subplots[2]
    sns.violinplot(
        y="y", x="charge", data=data, color="grey", inner=None, linewidth=0.75
    )
    axes.set(ylabel="DMS score", xlabel="579-588 charge")
    # axes.scatter(data['charge'], data['y'], s=5, alpha=0.2, c='black', lw=0)

    fig.tight_layout(h_pad=1.5, pad=0.5)
    fig.savefig("figures/aav_validations.png", dpi=300)
    fig.savefig("figures/aav_validations.svg", dpi=300)
