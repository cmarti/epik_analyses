#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from scripts.settings import AAV, RESULTSDIR
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mutation_decay_rates,
    savefig,
)

if __name__ == "__main__":
    dataset = AAV
    pos = "576"

    print("Loading {} decay factors for position {}".format(dataset, pos))
    fname = "{}.general_product_decay_rates.{}.csv".format(dataset, pos)
    df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)

    print("Plotting mutation specific decay rates for position {}".format(pos))
    fig, axes = plt.subplots(
        1, 1, figsize=(0.452 * FIG_WIDTH, 0.452 * FIG_WIDTH)
    )
    plot_mutation_decay_rates(
        axes, df, position=pos, dataset=dataset, cbar=False
    )
    sns.despine(ax=fig.axes[-1], right=False, top=False)

    fig.subplots_adjust(top=0.95, bottom=0.05)
    savefig(fig, "{}.decay_factors_576".format(dataset), dpi=300)
