#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.figures.utils import (
    load_decay_rates,
)
from scripts.figures.settings import ALPHABET
from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    dataset = "aav"
    alphabet = ALPHABET[dataset]
    general_product = load_decay_rates(dataset=dataset, kernel="GeneralProduct")

    # Mutation specific decay factors
    fig, axes = plt.subplots(
        1, 1, figsize=(0.452 * FIG_WIDTH, 0.452 * FIG_WIDTH)
    )
    position = "576"
    sns.heatmap(
        general_product[position] * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": r"Decay factor (%)", "shrink": 0.6},
        cbar=False,
    )
    ticks = np.arange(len(alphabet)) + 0.5
    axes.set(
        title="General Product model at {}".format(position),
        xticks=ticks,
        yticks=ticks,
        aspect="equal",
        xlabel="Allele 1",
        ylabel="Allele 2",
    )
    axes.set_yticklabels(alphabet, rotation=0, ha="center")
    axes.set_xticklabels(alphabet, rotation=0, ha="center")
    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=fig.axes[-1], right=False, top=False)

    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.savefig("figures/aav.gp_decay_factors_576.png", dpi=300)
    fig.savefig("figures/aav.gp_decay_factors_576.svg", dpi=300)
