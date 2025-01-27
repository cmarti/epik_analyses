#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"

    ena1_pos = 12
    data = pd.read_csv("datasets/qtls_li_hq.csv")
    data["ena1"] = [x[ena1_pos] for x in data["seq"]]
    data["nA"] = [x.count("A") for x in data["seq"]]
    wt = ["A" * 83, "B" * 83]

    bins = np.linspace(data["y"].min(), data["y"].max(), 50)
    fig, subplots = plt.subplots(1, 1, figsize=(2.9, 2.5))

    axes = subplots  # [0]
    idx = data["ena1"] == "A"
    axes.hist(
        data.loc[~idx, "y"], color="grey", label="ENA1-BY", alpha=0.5, bins=bins
    )
    axes.hist(
        data.loc[idx, "y"], color="black", label="ENA1-RM", alpha=0.5, bins=bins
    )
    axes.legend(loc=0, fontsize=8)
    axes.grid(alpha=0.2)
    axes.set(xlabel=r"Fitness", ylabel="# segregants")

    fig.tight_layout()
    fig.savefig("figures/yeast_validations.png", dpi=300)
    fig.savefig("figures/yeast_validations.svg", dpi=300)

    exit()

    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3))

    data["x"] = pd.cut(data["nA"], bins=np.linspace(10, 70, 7))
    data["label"] = "ENA1-RM"
    data.loc[data["ena1"] == "B", "label"] = "ENA1-BY"
    sns.boxplot(
        x="x",
        y="y",
        hue="label",
        data=data,
        ax=axes,
        # split=True,
        # inner="quartile",
        showfliers=False,
        linewidth=0.75,
    )
    axes.legend(loc=(0.0125, 1.05), ncol=2)
    axes.grid(alpha=0.2)
    axes.set(
        xlabel="Number of RM alleles",
        ylabel="Fitness",
        yticks=np.linspace(-0.5, 0, 6),
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig("figures/yeast_ena1_data.png", dpi=300)
