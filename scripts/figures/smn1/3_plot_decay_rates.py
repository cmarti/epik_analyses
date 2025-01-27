#!/usr/bin/env python
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.figures.plot_utils import highlight_seq_heatmap, FIG_WIDTH
from scripts.figures.utils import load_decay_rates

if __name__ == "__main__":
    dataset = "smn1"

    connectedness = load_decay_rates(dataset, kernel="Connectedness")
    jenga = load_decay_rates(dataset, kernel="Jenga")
    jenga.loc["+2", ["A", "G"]] = np.nan
    print(jenga)

    fig, subplots = plt.subplots(
        2,
        1,
        figsize=(0.375 * FIG_WIDTH, 0.25 * FIG_WIDTH),
        height_ratios=(1, 4),
        gridspec_kw={"hspace": 0.45},
    )
    cmap = cm.get_cmap("binary")
    cbar_axes = fig.add_axes([0.80, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.77, left=0.12, bottom=0.2)

    axes = subplots[0]
    axes.set_facecolor(cmap(0.1))
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
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    sns.despine(ax=axes, right=False, top=False)

    axes = subplots[1]
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(
        jenga.T * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_ax=cbar_axes,
        cbar_kws={"label": r"Decay factor (%)"},
    )
    axes.set(title="Jenga", xlabel="Position", ylabel="Allele")
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(axes, jenga, dataset)

    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)

    # fig.tight_layout()
    fig.savefig("figures/smn1.decay_factors.png", dpi=300)
    fig.savefig("figures/smn1.decay_factors.svg", dpi=300)
