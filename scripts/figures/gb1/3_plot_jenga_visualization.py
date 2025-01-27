#!/usr/bin/env python
import pandas as pd

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as plot

from gpmap.src.utils import read_edges
from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    kernel_visualization = "Jenga"
    seqs = ["VDGV", "WWLG", "LICA"]
    nrows, ncols = 1, len(seqs)

    # Load kernel function
    kernel = pd.read_csv("output/gb1.kernels_at_peaks.csv", index_col=0)

    # Load visualization
    nodes = pd.read_csv(
        "output/gb1.{}.nodes.csv".format(kernel_visualization), index_col=0
    )
    edges = read_edges("output/gb1.edges.npz")
    nodes = nodes.join(kernel)

    fig, subplots = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(0.6 * FIG_WIDTH, 0.22 * FIG_WIDTH),
        width_ratios=(1, 1, 1, 0.05),
    )
    cbar_ax = subplots[-1]

    for seq, axes in zip(seqs, subplots):
        nodes_color = "{}_{}".format(kernel_visualization, seq)
        max_cov = nodes[nodes_color].max()
        nodes[nodes_color] = nodes[nodes_color] / max_cov
        unique_covs = nodes[nodes_color].unique()

        plot.plot_visualization(
            axes,
            nodes,
            edges_df=edges,
            edges_alpha=0.01,
            nodes_vmax=1,
            nodes_vmin=0,
            nodes_cbar=True,
            nodes_cbar_axes=cbar_ax,
            nodes_cmap_label="K($x_1$, $x_2$)",
            nodes_color=nodes_color,
            nodes_cmap="viridis",
            sort_by=nodes_color,
            sort_ascending=True,
            fontsize=8,
        )

        axes.set(xlabel="", ylabel="", aspect="equal")
        axes.text(
            0.95,
            0.95,
            seq,
            transform=axes.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        if seq != seqs[0]:
            axes.set(yticklabels=[])
        axes.grid(alpha=0.2, lw=0.5)

    subplots[0].set_ylabel("Diffusion axis 2", fontsize=8)
    subplots[1].set_xlabel("Diffusion axis 1", fontsize=8)

    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/gb1.visualization.kernels_main.png", dpi=300)
