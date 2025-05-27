#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os.path import join
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mutation_decay_rates,
    savefig,
)
from scripts.settings import GB1, RESULTSDIR


if __name__ == "__main__":
    dataset = GB1
    figsize = (0.325 * FIG_WIDTH, 0.65 * FIG_WIDTH)
    fig, subplots = plt.subplots(2, 1, figsize=figsize, sharex=True)
    cbar_axes = fig.add_axes([0.65, 0.3, 0.04, 0.4])

    pos = "41"
    print("Loading mutation decay rates for position {}".format(pos))
    fname = "{}.GeneralProduct_decay_rates.{}.csv".format(dataset, pos)
    df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)

    print("Plotting mutation decay rates for position {}".format(pos))
    axes = subplots[0]
    plot_mutation_decay_rates(
        axes, df, position=pos, dataset=dataset, cbar=False
    )
    axes.set(
        title="Position {}".format(pos),
        xlabel="",
        xticklabels=[],
    )

    pos = '54'
    print("Loading mutation decay rates for position {}".format(pos))
    fname = "{}.GeneralProduct_decay_rates.{}.csv".format(dataset, pos)
    df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)

    print("Plotting mutation decay rates for position {}".format(pos))
    axes = subplots[1]
    plot_mutation_decay_rates(
        axes, df, position=pos, dataset=dataset, cbar_ax=cbar_axes
    )
    axes.set(title="Position {}".format(pos))

    sns.despine(ax=cbar_axes, right=False, top=False)
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.15, right=0.9, hspace=0.1)
    savefig(fig, "gb1.decay_factors.general_product")
