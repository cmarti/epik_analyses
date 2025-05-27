#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from scripts.settings import AAV, RESULTSDIR
from scripts.figures.plot_utils import FIG_WIDTH, plot_decay_rates, savefig


if __name__ == "__main__":
    dataset = AAV

    print("Loading {} decay factors".format(dataset))
    fname = "{}.connectedness_decay_rates.csv".format(dataset)
    connectedness = pd.read_csv(join(RESULTSDIR, fname), index_col=0)

    fpath = join(RESULTSDIR, "{}.jenga_decay_rates.csv".format(dataset))
    jenga = pd.read_csv(fpath, index_col=0)
    print(jenga)

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
    plot_decay_rates(axes, connectedness, dataset, cbar=False)
    axes.set(
        title="Connectedness model",
        yticks=[],
        xticklabels=[],
        xlabel="",
        ylabel="",
    )

    axes = subplots[1]
    plot_decay_rates(axes, jenga, dataset, cbar_ax=cbar_axes)
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    sns.despine(ax=cbar_axes, right=False, top=False)
    fig.subplots_adjust(bottom=0.15, top=0.95)
    savefig(fig, "{}.decay_factors".format(dataset))
