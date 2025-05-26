#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
import gpmap.plot.mpl as mplot


def annotate_seq(
    axes,
    seq,
    label,
    df,
    dx,
    dy,
    ha,
    va,
    x="1",
    y="2",
    fontsize=8,
    arrow_size=0.75,
):
    x, y = df.loc[seq, [x, y]]
    axes.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=0.3 * arrow_size,
            headwidth=4 * arrow_size,
            headlength=6 * arrow_size,
        ),
        ha=ha,
        va=va,
        fontsize=fontsize,
    )


if __name__ == "__main__":
    wt_seq = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
    fpath = "output_new/aav.GeneralProduct.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)
    seqs = np.load("datasets/aav.pred.labels.npy", allow_pickle=True)
    data = pd.read_csv("datasets/aav.csv", index_col=0)
    data["subseq"] = [x[6:10] for x in data.index]
    print(data.loc[[x[0] == "E" for x in data["subseq"]], :])
    print(data.sort_values("y").tail(50))

    space = SequenceSpace(X=seqs, y=landscape["coef"].values)
    edges = space.get_edges_df()

    # print("Calculating visualization")
    # rw = WMWalk(space)
    # rw.calc_visualization(mean_function=-0.1, n_components=20)
    # rw.nodes_df.to_parquet("results/aav.nodes2.pq")
    # nodes = rw.nodes_df

    print("Loading visualization")
    nodes = pd.read_parquet("results/aav.nodes2.pq")
    print(nodes.sort_values("1").head(20))
    print(nodes.sort_values("1").tail(20))

    print(nodes.sort_values("2").head(20))
    print(nodes.sort_values("2").tail(20))
    # exit()

    fig, axes = plt.subplots(1, 1, figsize=(4.5, 3.5))
    mplot.plot_visualization(
        axes,
        nodes,
        nodes_size=2.5,
        edges_df=edges,
    )
    # axes.set(aspect="equal", ylim=(-2.5, 2.5), xlim=(-2.0, 7.5))
    annotate_seq(
        axes,
        "TTNP",
        "WT",
        nodes,
        dx=0,
        dy=-0.5,
        ha="center",
        va="top",
    )

    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/aav_visualization.png", dpi=300)
