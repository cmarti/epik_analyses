#!/usr/bin/env python
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
    fpath = "output_new/qtls_li_hq.Connectedness.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)

    with open("datasets/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]

    landscape.index = seqs

    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=-0.1, n_components=20)
    nodes = rw.nodes_df
    edges = space.get_edges_df()

    fig, axes = plt.subplots(1, 1, figsize=(4.5, 2.5))
    mplot.plot_visualization(
        axes,
        nodes,
        nodes_size=2.5,
        edges_df=edges,
    )
    axes.set(aspect="equal", ylim=(-2.5, 2.5), xlim=(-2.0, 7.5))

    seqs = ["A" * 17, "B" * 17, "BBAAAABAAABABABBB"]
    df = nodes.loc[seqs, :]
    axes.scatter(
        df["1"],
        df["2"],
        c=df["function"],
        cmap="viridis",
        s=2.5,
        lw=1,
        edgecolors="black",
    )

    annotate_seq(
        axes,
        "A" * 17,
        "RM",
        nodes,
        dx=1.25,
        dy=1.25,
        ha="left",
        va="bottom",
    )

    annotate_seq(
        axes,
        "B" * 17,
        "BY",
        nodes,
        dx=-1.25,
        dy=-1.25,
        ha="right",
        va="top",
    )
    annotate_seq(
        axes,
        "BBAAAABAAABABABBB",
        "Best ENA1-RM",
        nodes,
        dx=0.5,
        dy=-0.5,
        ha="center",
        va="top",
    )

    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/yeast_visualization.png", dpi=300)
