#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scripts.figures.utils import load_decay_rates
from scripts.figures.plot_utils import highlight_seq_heatmap, FIG_WIDTH
from scripts.figures.settings import ALPHABET


if __name__ == "__main__":
    dataset = "gb1"
    alphabet = ALPHABET[dataset]

    # exponential = load_decay_rates(dataset=dataset, kernel="Exponential")
    connectedness = load_decay_rates(dataset=dataset, kernel="Connectedness")
    jenga = load_decay_rates(dataset=dataset, kernel="Jenga")

    # Site and allele specific decay factors
    figsize = (FIG_WIDTH * 0.165, FIG_WIDTH * 0.425)
    fig, subplots = plt.subplots(
        2,
        1,
        figsize=figsize,
        height_ratios=(1, 18),
        # gridspec_kw={"hspace": 0.225},
    )
    # cbar_axes = fig.add_axes([0.65, 0.3, 0.04, 0.4])
    # fig.subplots_adjust(right=0.60, left=0.25)

    # Connectedness model
    axes = subplots[0]
    sns.heatmap(
        connectedness.T * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=False,
    )
    axes.set(
        title="Connectedness",
        yticks=[],
        xticklabels=[],
        xlabel="",
        ylabel="",
    )
    axes.set_yticks([])
    sns.despine(ax=axes, right=False, top=False)

    # Jenga model
    axes = subplots[1]
    sns.heatmap(
        jenga.T * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        # cbar_ax=cbar_axes,
        cbar=False,
        cbar_kws={"label": r"Decay factor (%)"},
    )
    axes.set(
        title="Jenga",
        xlabel="Position",
        ylabel="Allele",
        xticks=np.arange(4) + 0.5,
        yticks=np.arange(len(alphabet)) + 0.5,
    )
    axes.set_yticklabels(alphabet, rotation=0, ha="center", fontsize=7)
    axes.set_xticklabels(jenga.index, rotation=0, ha="center", fontsize=7)
    highlight_seq_heatmap(axes, jenga, dataset=dataset)
    sns.despine(ax=axes, right=False, top=False)

    # sns.despine(ax=cbar_axes, right=False, top=False)
    fig.tight_layout(h_pad=0.3)
    fig.savefig("figures/gb1.decay_factors.png", dpi=300)
    fig.savefig("figures/gb1.decay_factors.svg", dpi=300)
