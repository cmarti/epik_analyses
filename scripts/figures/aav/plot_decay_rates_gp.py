#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.figures.utils import load_decay_rates
from scripts.figures.settings import ALPHABET


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    dataset = "aav"
    alphabet = ALPHABET[dataset]

    general_product = load_decay_rates(
        dataset=dataset, kernel="GeneralProduct", id=50
    )

    # Mutation specific decay factors
    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(7.5, 3.75),
        gridspec_kw={"wspace": 0.1},
        width_ratios=(1, 1, 0.05),
    )
    cbar_axes = subplots[-1]

    axes = subplots[0]
    pos = "569"
    df = general_product[pos] * 100
    sns.heatmap(
        df,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=False,
    )
    ticks = np.arange(len(alphabet)) + 0.5
    axes.set(
        title="Position {}".format(pos),
        ylabel="Allele 2",
        xlabel="Allele 1",
        xticks=ticks,
        yticks=ticks,
        xticklabels=alphabet,
    )
    axes.set_yticklabels(alphabet, rotation=0)
    sns.despine(ax=axes, right=False, top=False)

    axes = subplots[1]
    pos = "576"
    df = general_product[pos] * 100
    sns.heatmap(
        df,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_ax=cbar_axes,
        cbar_kws={"label": r"Decay factor (%)"},
    )
    axes.set(
        title="Position {}".format(pos),
        xlabel="Allele 1",
        xticks=ticks,
        yticks=ticks,
        xticklabels=alphabet,
    )
    axes.set_yticklabels([])

    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)

    # fig.tight_layout()
    fig.savefig("figures/aav.decay_factors.general_product.png", dpi=300)
    fig.savefig("figures/aav.decay_factors.general_product.svg", dpi=300)

    for pos, m in general_product.items():
        fig = sns.clustermap(m, cmap="Blues", vmin=0, vmax=1)
        fig.savefig("plots/aav.general_product.pos{}.png".format(pos), dpi=300)
