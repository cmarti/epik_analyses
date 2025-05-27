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
from scripts.settings import POSITIONS, RESULTSDIR, GB1


if __name__ == "__main__":
    dataset = GB1
    positions = POSITIONS[dataset]

    figsize = (0.8 * FIG_WIDTH, 1.3 * FIG_WIDTH)
    fig, subplots = plt.subplots(4, 2, figsize=figsize, sharex=True)
    cbar_axes = fig.add_axes([0.875, 0.4, 0.015, 0.2])
    fig.subplots_adjust(
        right=0.85, left=0.1, hspace=0.25, wspace=0.25, top=0.9, bottom=0.1
    )

    print("Plotting mutation specific decay rates in {}".format(dataset))
    models = ["jenga", "general_product"]
    for axes_row, pos in zip(subplots, positions):
        
        print('\tPosition {}'.format(pos))
        for axes, model in zip(axes_row, models):
            fname = "{}.{}_decay_rates.{}.csv".format(dataset, model, pos)
            df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)
            plot_mutation_decay_rates(
                axes, df, position=pos, dataset=dataset, cbar_ax=cbar_axes
            )
            title = "{} model; Position {}"
            title = title.format(model.replace("_", " ").capitalize(), pos)
            axes.set(title=title)

    sns.despine(ax=cbar_axes, right=False, top=False)
    fig.subplots_adjust(top=0.975, bottom=0.05, left=0.1)

    savefig(fig, "gb1.decay_factors_supp")
