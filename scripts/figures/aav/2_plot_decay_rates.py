#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.figures.plot_utils import FIG_WIDTH, highlight_seq_heatmap
from scripts.figures.utils import load_decay_rates


if __name__ == "__main__":
    dataset = "aav"

    connectedness = load_decay_rates(dataset=dataset, kernel="Connectedness")
    jenga = load_decay_rates(dataset=dataset, kernel="Jenga")

    fig, subplots = plt.subplots(
        2,
        1,
        figsize=(0.625 * FIG_WIDTH, 0.51 * FIG_WIDTH),
        height_ratios=(1, 16),
        gridspec_kw={"hspace": 0.2},
    )
    cbar_axes = fig.add_axes([0.875, 0.3, 0.02, 0.4])
    fig.subplots_adjust(right=0.85, left=0.1)

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
        title="Connectedness model",
        yticks=[],
        xticklabels=[],
        xlabel="",
        ylabel="",
    )
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    sns.despine(ax=axes, right=False, top=False)

    axes = subplots[1]
    sns.heatmap(
        jenga.T * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_ax=cbar_axes,
        cbar_kws={"label": r"Decay factor (%)"},
    )
    axes.set(title="Jenga model", xlabel="Position", ylabel="Allele")
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(axes, jenga, dataset=dataset)

    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)

    fig.subplots_adjust(bottom=0.15, top=0.95)
    fig.savefig("figures/aav.decay_factors.png", dpi=300)
    fig.savefig("figures/aav.decay_factors.svg", dpi=300)
