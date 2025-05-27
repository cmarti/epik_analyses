#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from scripts.settings import POSITIONS, AAV, RESULTSDIR
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_mutation_decay_rates,
    savefig,
)

if __name__ == "__main__":
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    
    dataset = AAV
    positions = POSITIONS[dataset]
    figsize = (FIG_WIDTH, 1.75 * FIG_WIDTH)
    
    # Mutation specific decay factors
    fig, subplots = plt.subplots(7, 4, figsize=(10, 15))
    cbar_axes = fig.add_axes([0.925, 0.425, 0.0075, 0.15])
    fig.subplots_adjust(
        right=0.90, left=0.05, hspace=0.25, wspace=0.25, top=0.95, bottom=0.15
    )
    subplots = subplots.flatten()

    print("Plotting mutation specific decay rates in {}".format(dataset))
    for axes, pos in zip(subplots, positions):
        print("\tPosition {}".format(pos))
        fname = "{}.general_product_decay_rates.{}.csv".format(dataset, pos)
        df = pd.read_csv(join(RESULTSDIR, fname), index_col=0)
        plot_mutation_decay_rates(
            axes, df, position=pos, dataset=dataset, cbar_ax=cbar_axes
        )
        axes.set(xlabel='', ylabel='')

    sns.despine(ax=cbar_axes, right=False, top=False)
    fig.supxlabel("Allele 1", fontsize=8)
    fig.supylabel("Allele 2", fontsize=8)

    fig.subplots_adjust(top=0.975, bottom=0.035, hspace=0.25)
    savefig(fig, "{}.decay_factors_supp".format(dataset))
