#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from matplotlib import colormaps
from scripts.settings import SMN1, RESULTSDIR
from scripts.figures.plot_utils import FIG_WIDTH, plot_decay_rates, savefig


if __name__ == "__main__":
    dataset = SMN1

    print("Loading {} decay factors".format(dataset))
    fname = "{}.connectedness_decay_rates.csv".format(dataset)
    connectedness = pd.read_csv(join(RESULTSDIR, fname), index_col=0)

    fpath = join(RESULTSDIR, "{}.jenga_decay_rates.csv".format(dataset))
    jenga = pd.read_csv(fpath, index_col=0)

    print("Plotting {} decay factors".format(dataset))
    fig, subplots = plt.subplots(
        2,
        1,
        figsize=(0.375 * FIG_WIDTH, 0.25 * FIG_WIDTH),
        height_ratios=(1, 4),
        gridspec_kw={"hspace": 0.45},
    )
    cmap = colormaps["binary"]
    cbar_axes = fig.add_axes([0.80, 0.275, 0.03, 0.5])
    fig.subplots_adjust(right=0.77, left=0.12, bottom=0.2)

    axes = subplots[0]
    axes.set_facecolor(cmap(0.1))
    plot_decay_rates(axes, connectedness, dataset, cbar=False)
    axes.set(
        title="Connectedness",
        yticks=[],
        xticklabels=[],
        xlabel="",
        ylabel="",
    )

    axes = subplots[1]
    axes.set_facecolor(cmap(0.1))
    plot_decay_rates(axes, jenga, dataset, cbar_ax=cbar_axes)

    sns.despine(ax=cbar_axes, right=False, top=False)
    savefig(fig, "{}.decay_factors".format(dataset))
