#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.figures.utils import (
    load_decay_rates,
    get_jenga_mut_decay_rates,
)
from scripts.figures.plot_utils import FIG_WIDTH
from scripts.figures.settings import ALPHABET



if __name__ == "__main__":
    dataset = "aav"
    alphabet = ALPHABET[dataset]
    positions = [569, 576, 585, 588]
    # positions = POSITIONS[dataset]
    general_product = load_decay_rates(dataset=dataset, kernel="GeneralProduct")
    jenga = load_decay_rates(dataset=dataset, kernel="Jenga")
    jenga = get_jenga_mut_decay_rates(jenga)
    dfs = {"Jenga": jenga, "General Product": general_product}

    # Mutation specific decay factors
    figsize = (0.8 * FIG_WIDTH, 1.3 * FIG_WIDTH)
    fig, subplots = plt.subplots(
        len(positions), 2, figsize=figsize, sharex=True
    )
    cbar_axes = fig.add_axes([0.875, 0.4, 0.015, 0.2])
    fig.subplots_adjust(
        right=0.85, left=0.1, hspace=0.25, wspace=0.25, top=0.9, bottom=0.1
    )

    for axes_row, pos in zip(subplots, positions):
        for axes, (model, decay_rates) in zip(axes_row, dfs.items()):
            sns.heatmap(
                decay_rates[str(pos)] * 100,
                ax=axes,
                cmap="Blues",
                vmin=0,
                vmax=100,
                cbar_ax=cbar_axes,
                cbar_kws={"label": r"Decay factor (%)"},
            )
            ticks = np.arange(len(alphabet)) + 0.5
            axes.set(
                title="{} Model; Position {}".format(model, pos),
                ylabel="Allele 2",
                xlabel="Allele 1",
                xticks=ticks,
                yticks=ticks,
                aspect="equal",
            )
            axes.set_yticklabels(alphabet, rotation=0, ha="center", fontsize=7)
            axes.set_xticklabels(alphabet, rotation=0, ha="center", fontsize=7)
            sns.despine(ax=axes, right=False, top=False)

    sns.despine(ax=cbar_axes, right=False, top=False)

    # fig.tight_layout(h_pad=0.5)
    fig.subplots_adjust(top=0.95, bottom=0.05)
    fig.savefig("figures/aav.decay_factors_supp.png", dpi=300)
    fig.savefig("figures/aav.decay_factors_supp.svg", dpi=300)
