#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from matplotlib import colormaps
from scripts.settings import POSITIONS, SMN1, RESULTSDIR
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mutation_decay_rates,
    savefig,
)


if __name__ == "__main__":
    dataset = SMN1
    positions = POSITIONS[dataset]

    figsize = (FIG_WIDTH, 0.25 * FIG_WIDTH)
    fig, subplots = plt.subplots(
        2, len(positions), figsize=figsize, sharex=True, sharey=True
    )
    subplots = subplots.T
    fig.subplots_adjust(
        right=0.90, left=0.11, hspace=0.0, wspace=0.5, top=0.9, bottom=0.175
    )
    cbar_axes = fig.add_axes([0.915, 0.325, 0.01, 0.4])

    print("Plotting mutation specific decay rates in {}".format(dataset))
    bg_color = colormaps["binary"](0.1)
    models = ["jenga", "general_product"]
    for axes_row, pos in zip(subplots, positions):
        print("\tPosition {}".format(pos))
        for axes, model in zip(axes_row, models):
            axes.set_facecolor(bg_color)

            fname = "{}.{}_decay_rates.{}.csv".format(dataset, model, pos)
            df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)
            plot_mutation_decay_rates(
                axes, df, position=pos, dataset=dataset, cbar_ax=cbar_axes
            )
            title = "Position {}".format(pos) if model == "jenga" else ""
            axes.set(xlabel="", ylabel="", title=title)

    sns.despine(ax=cbar_axes, right=False, top=False)
    subplots[0][0].set(ylabel="Jenga\nmodel")
    subplots[0][1].set(ylabel="General product\nmodel")
    fig.supxlabel("Allele 1", fontsize=8)
    fig.supylabel("Allele 2", fontsize=8)

    savefig(fig, "{}.decay_factors_supp".format(dataset))
