#!/usr/bin/env python
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.settings import AAV, RESULTSDIR, AAV_BACKGROUNDS, AMINOACIDS
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    highlight_seq_heatmap,
    savefig,
)


if __name__ == "__main__":
    dataset = AAV
    figsize = (0.55 * FIG_WIDTH, 0.5 * FIG_WIDTH)

    labels = ["WT", "N587E", "R585E+N587E"]
    fpath = join(RESULTSDIR, "{}.mutational_effects.csv".format(dataset))
    mut_effs = pd.read_csv(fpath, index_col=0)

    print("Plotting heatmap of predicted mutational effects")
    for label in labels:
        print("\tIn {} background".format(label))
        col = "coef_{}".format(label)
        df = mut_effs.dropna(subset=col)
        df = pd.pivot_table(
            df, index="allele2", columns="position", values=col
        )
        df = df.fillna(0).loc[AMINOACIDS, :]

        fig, axes = plt.subplots(1, 1, figsize=figsize)
        cbar_axes = fig.add_axes([0.875, 0.3, 0.02, 0.4])
        fig.subplots_adjust(right=0.85, left=0.1, bottom=0.15)

        sns.heatmap(
            df,
            ax=axes,
            cmap="coolwarm",
            center=0,
            cbar_ax=cbar_axes,
            vmin=-8,
            vmax=3,
            cbar_kws={"label": "Mutational effect"},
        )
        axes.set(
            xlabel="Position",
            ylabel="Allele",
            aspect="equal",
            xticks=0.5 + np.arange(28),
        )
        axes.set_yticklabels(axes.get_yticklabels(), rotation=0, fontsize=8)
        axes.set_xticklabels(df.columns, rotation=90, fontsize=8)
        highlight_seq_heatmap(axes, df.T, seq=AAV_BACKGROUNDS[label])

        sns.despine(right=False, top=False)
        savefig(fig, "aav.mut_effs_{}".format(label))
