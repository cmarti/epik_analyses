#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gpmap.space import SequenceSpace
import gpmap.plot.mpl as mplot

from scripts.figures.plot_utils import FIG_WIDTH


def main():
    # Load data
    fpath = "output/models/qtls_li_hq.Connectedness.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)

    loci = np.load("data/processed/qtls_li_hq.selected_loci.npy", allow_pickle=True)
    loci = np.append(["BC"], loci)

    with open("data/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]

    # Process data
    landscape.index = seqs
    ena1_idx = list(loci).index("ENA1")
    landscape["ena1"] = [seq[ena1_idx] for seq in landscape.index]
    best_ena1_rm = landscape.loc[landscape["ena1"] == "A", "coef"].idxmax()
    landscape["d"] = [
        sum(c1 != c2 for c1, c2 in zip(seq, best_ena1_rm))
        for seq in landscape.index
    ]
    landscape["d"] += np.random.normal(0, 0.1, size=landscape.shape[0])

    # Create sequence space and edges
    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    edges = space.get_edges_df()
    palette = {"B": "grey", "A": "black"}

    # Plot visualization
    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(0.8 * FIG_WIDTH, 0.4 * FIG_WIDTH),
        width_ratios=[1, 0.5],
        sharey=True,
    )
    ax = subplots[0]
    mplot.plot_visualization(
        ax,
        landscape,
        x="d",
        y="coef",
        nodes_size=5,
        # edges_df=edges,
        nodes_color="ena1",
        nodes_palette=palette,
        nodes_alpha=0.1,
        edges_color="lightgrey",
        rasterized=True,
    )
    ax.set(xlabel="Hamming distance to best ENA1-RM", ylabel="Fitness")
    ax.legend_.set_visible(False)

    ax = subplots[1]
    bins = np.linspace(landscape["coef"].min(), landscape["coef"].max(), 50)
    ax.hist(
        landscape.loc[landscape["ena1"] == "B", "coef"],
        bins=bins,
        color=palette["B"],
        alpha=0.5,
        orientation="horizontal",
        label="ENA1-BY",
    )
    ax.hist(
        landscape.loc[landscape["ena1"] == "A", "coef"],
        bins=bins,
        color=palette["A"],
        alpha=0.5,
        orientation="horizontal",
        label="ENA1-RM",
    )
    ax.legend(loc=4)
    ax.set(xlabel="# genotypes")

    fig.tight_layout(w_pad=1)
    fig.savefig("figures/david_yeast_visualization.png", dpi=300)
    fig.savefig("figures/david_yeast_visualization.svg", dpi=300)


if __name__ == "__main__":
    main()
