#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
import gpmap.plot.mpl as mplot
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    fpath = "output_new/qtls_li_hq.Connectedness.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)

    with open("datasets/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]
    seq_length = len(seqs[0])

    landscape.index = seqs
    cols = np.append(
        ["BC"],
        np.load("datasets/qtls_li_hq.selected_loci.npy", allow_pickle=True),
    )

    fig, subplots = plt.subplots(
        2, 4, figsize=(FIG_WIDTH, 0.5 * FIG_WIDTH), sharex=True, sharey=True
    )
    subplots = subplots.flatten()

    for i, (axes, locus) in enumerate(zip(subplots, cols)):
        alleles = []
        bc = []
        for seq in seqs:
            alleles.append(seq[i])
            bc.append("".join(a for j, a in enumerate(seq) if j != i))
        landscape["bc"] = bc
        landscape["allele"] = alleles
        means = pd.pivot_table(
            landscape, index="bc", columns="allele", values="coef"
        )
        err = 2 * pd.pivot_table(
            landscape, index="bc", columns="allele", values="stderr"
        )
        axes.axline((0, 0), slope=1, color="grey", linestyle="--", lw=0.5)

        axes.errorbar(
            means["A"],
            means["B"],
            xerr=err["A"],
            yerr=err["B"],
            fmt="o",
            color="black",
            alpha=0.5,
            markersize=1,
            lw=0.3,
        )

        space = SequenceSpace(X=means.index.values, y=means["A"].values)
        edf = space.get_edges_df()
        # mplot.plot_edges(
        #     axes,
        #     means,
        #     edf,
        #     x="A",
        #     y="B",
        #     color="grey",
        #     alpha=0.5,
        #     width=0.2,
        # )

        seq1 = "A" * (seq_length - 1)
        axes.errorbar(
            means.loc[seq1, "A"],
            means.loc[seq1, "B"],
            xerr=err.loc[seq1, "A"],
            yerr=err.loc[seq1, "B"],
            fmt="o",
            color="red",
            alpha=1,
            markersize=2,
            lw=0.5,
        )

        seq2 = "B" * (seq_length - 1)
        axes.errorbar(
            means.loc[seq2, "A"],
            means.loc[seq2, "B"],
            xerr=err.loc[seq2, "A"],
            yerr=err.loc[seq2, "B"],
            fmt="o",
            color="red",
            alpha=1,
            markersize=2,
            lw=0.5,
        )

        axes.text(
            0.05,
            0.95,
            locus,
            transform=axes.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )
    fig.supxlabel("Fitness in RM background", fontsize=9)
    fig.supylabel("Fitness in BY background", fontsize=9)

    fig.tight_layout()
    fig.savefig("figures/yeast_landscape_scatter.png", dpi=300)
