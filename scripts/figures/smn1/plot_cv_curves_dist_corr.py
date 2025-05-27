#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpmap.plot.mpl as plot

from os.path import join
from gpmap.space import SequenceSpace
from scripts.settings import SMN1, RESULTSDIR
from scripts.utils import load_r2
from scripts.figures.plot_utils import FIG_WIDTH, plot_cv_curve, savefig


def plot_vj_correlations(axes, covs):
    covs["x"] = np.random.normal(covs["d"], scale=0.075)
    space = SequenceSpace(X=covs.index.values, y=covs["data"].values)
    edges = space.get_edges_df()

    plot.plot_visualization(
        axes,
        covs,
        edges_df=edges,
        x="x",
        y="data",
        nodes_size=6.5,
        nodes_color="grey",
        nodes_alpha=0.6,
        edges_color="lightgrey",
        edges_alpha=0.4,
    )

    df = covs[["d", "data", "data_ns"]].copy()
    df.columns = ["d", "c", "n"]
    df["c"] = df["c"] * df["n"]
    df = df.groupby(["d"])[["c", "n"]].sum().reset_index()
    df["c"] = df["c"] / df["n"]

    axes.plot(df["d"], df["c"], lw=1, color="black", zorder=5)
    axes.scatter(df["d"], df["c"], s=10, color="black", zorder=5)

    lim = (-0.1, 1.05)
    axes.set(
        xticks=np.arange(space.seq_length + 1),
        ylim=lim,
        xlim=(-0.5, 8.5),
        aspect=9 / (lim[1] - lim[0]),
    )
    axes.set_xlabel("Hamming distance", fontsize=8)
    axes.set_ylabel("Correlation", fontsize=8)


if __name__ == "__main__":
    dataset = SMN1
    metric = "r2"

    print("Loading {} empirical covariances".format(dataset))
    fpath = join(RESULTSDIR, "{}.vj_covariances.csv".format(dataset))
    covs = pd.read_csv(fpath, index_col=0).sort_index()
    data = load_r2(dataset)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(FIG_WIDTH * 0.7, FIG_WIDTH * 0.4),
    )

    axes = subplots[0]
    plot_cv_curve(axes, data, metric=metric)

    axes = subplots[1]
    plot_vj_correlations(axes, covs)

    fig.tight_layout()
    savefig(fig, "smn1_r2_distance_correlations")
