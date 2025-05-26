#!/usr/bin/env python
import pandas as pd
import numpy as np

import gpmap.src.plot.mpl as mplot
import gpmap.src.plot.ds as dplot
import holoviews as hv

from gpmap.src.utils import read_edges
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    hv.extension("matplotlib")

    kernel_visualization = "Jenga"
    labels = [
        "Additive",
        "Pairwise",
        "Variance Component",
        "Exponential",
        "Connectedness",
        "Jenga",
        "GeneralProduct",
    ]
    seqs = ["VDGV", "WWLG", "LICA"]
    nrows, ncols = len(seqs), len(labels)

    # Load kernel function
    kernel = pd.read_csv("output_new/gb1.kernels_at_peaks.csv", index_col=0)

    # Load visualization
    nodes = pd.read_csv("output_new/gb1.jenga.nodes.csv", index_col=0)
    edges = read_edges("output/gb1.edges.npz")
    nodes = nodes.join(kernel)

    edges_dsg = dplot.plot_edges(nodes, edges_df=edges, resolution=1200).opts(
        padding=0.05
    )
    dsg = edges_dsg
    for i in range(20):
        dsg += edges_dsg
    grid = hv.Layout(dsg).cols(3)
    grid.opts(sublabel_format="")
    fig = dplot.dsg_to_fig(grid)
    fig.set_size_inches(FIG_WIDTH, 1.6 * FIG_WIDTH)
    subplots = np.array(fig.axes).reshape((7, 3))
    print(subplots)

    # fig, subplots = plt.subplots(
    #     ncols,
    #     nrows,
    #     figsize=(FIG_WIDTH, 1.6 * FIG_WIDTH),
    #     sharex=True,
    #     sharey=True,
    # )

    for label, axes_row in zip(labels, subplots):
        print("Plotting kernel {}".format(label))
        for seq, axes in zip(seqs, axes_row):
            nodes_color = "{}_{}".format(label, seq)
            max_cov = nodes[nodes_color].max()
            nodes[nodes_color] = nodes[nodes_color] / max_cov
            unique_covs = nodes[nodes_color].unique()
            # print("\t", seq, unique_covs.shape, max_cov)
            if unique_covs.shape[0] < 20:
                print(unique_covs)
            mplot.plot_visualization(
                axes,
                nodes,
                # edges_df=edges,
                edges_alpha=0.05,
                nodes_vmax=1,
                nodes_size=1,
                nodes_vmin=0,
                nodes_cbar=True,
                nodes_cmap_label="K({}, x)".format(seq),
                nodes_color=nodes_color,
                nodes_cmap="viridis",
                sort_by=nodes_color,
                sort_ascending=True,
            )
            axes.set(xlabel="", ylabel="", aspect="equal")
            axes.text(
                0.98,
                0.98,
                label,
                transform=axes.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
    for axes in subplots[-1]:
        axes.set(xlabel="Diffusion axis 1")
    for axes in subplots[:, 0]:
        axes.set(ylabel="Diffusion axis 2")

    fig.tight_layout()
    fig.subplots_adjust(left=0.075)
    fig.savefig("figures/gb1.visualization.kernels.png", dpi=300)
