#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt

from os.path import join
from scripts.figures.plot_utils import (
    plot_decay_rates,
    FIG_WIDTH,
    savefig,
)
from scripts.settings import GB1, RESULTSDIR


if __name__ == "__main__":
    print("Loading GB1 site and allele specific decay rates from full model")
    dataset = GB1
    fpath = join(
        RESULTSDIR, "{}.connectedness_decay_rates.csv".format(dataset)
    )
    connectedness = pd.read_csv(fpath, index_col=0)

    fpath = join(RESULTSDIR, "{}.jenga_decay_rates.csv".format(dataset))
    jenga = pd.read_csv(fpath, index_col=0)

    # Site and allele specific decay factors
    figsize = (FIG_WIDTH * 0.165, FIG_WIDTH * 0.425)
    fig, subplots = plt.subplots(
        2,
        1,
        figsize=figsize,
        height_ratios=(1, 18),
    )

    print("Loading site specific decay factors from connectedness model")
    axes = subplots[0]
    plot_decay_rates(axes, connectedness, dataset, cbar=False)
    axes.set(
        title="Connectedness",
        yticks=[],
        xticklabels=[],
        xlabel="",
        ylabel="",
    )

    print("Loading allele specific decay factors from Jenga model")
    axes = subplots[1]
    plot_decay_rates(axes, jenga, dataset, cbar=False)

    fig.tight_layout(h_pad=0.3)
    savefig(fig, "gb1.decay_factors")
