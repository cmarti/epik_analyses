#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.figures.utils import (
    load_decay_rates,
)
from scripts.figures.settings import ALPHABET


if __name__ == "__main__":
    dataset = "aav"
    alphabet = ALPHABET[dataset]
    general_product = load_decay_rates(dataset=dataset,
                                       kernel="GeneralProduct")

    # Mutation specific decay factors
    figsize = (10, 17.5)
    fig, subplots = plt.subplots(7, 4, figsize=figsize,
                                #  sharex=True, sharey=True,
                                 )
    cbar_axes = fig.add_axes([0.925, 0.4, 0.0075, 0.2])
    fig.subplots_adjust(
        right=0.90, left=0.05, hspace=0.25, wspace=0.25, top=0.95, bottom=0.15
    )
    subplots = subplots.flatten()

    for axes, (position, decay_rates) in zip(subplots,
                                             general_product.items()):
        sns.heatmap(
            decay_rates * 100,
            ax=axes,
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar_ax=cbar_axes,
            cbar_kws={"label": r"Decay factor (%)"},
        )
        ticks = np.arange(len(alphabet)) + 0.5
        axes.set(
            title="Position {}".format(position),
            xticks=ticks,
            yticks=ticks,
            aspect="equal",
        )
        axes.set_yticklabels(alphabet, rotation=0, ha="center", fontsize=7)
        axes.set_xticklabels(alphabet, rotation=0, ha="center", fontsize=7)
        sns.despine(ax=axes, right=False, top=False)

    fig.supxlabel("Allele 1", fontsize=10)
    fig.supylabel("Allele 2", fontsize=10)
    sns.despine(ax=cbar_axes, right=False, top=False)

    # fig.tight_layout(h_pad=0.5)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.savefig("figures/aav.gp_decay_factors.png", dpi=300)
    # fig.savefig("figures/aav.gp_decay_factors.svg", dpi=300)
