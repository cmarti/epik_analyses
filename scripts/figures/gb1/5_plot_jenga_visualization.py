#!/usr/bin/env python
# import gpmap.src.plot.ds as dplot
import gpmap.src.plot.mpl as mplot

# import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpmap.src.utils import read_edges

from scripts.figures.plot_utils import FIG_WIDTH

if __name__ == "__main__":
    # hv.extension("matplotlib")
    seqs = ["VDGV", "WWLG", "LICA"]
    nrows, ncols = 1, len(seqs)

    # Load kernel function
    kernel = pd.read_csv("output_new/gb1.kernels_at_peaks.csv", index_col=0)

    # Load visualization
    nodes = pd.read_csv("output_new/gb1.jenga.nodes.csv", index_col=0)
    edges = read_edges("output_new/gb1.jenga.edges.npz")
    nodes = nodes.join(kernel)

    # edges_dsg = dplot.plot_edges(nodes, edges_df=edges, resolution=1200).opts(
    #     padding=0.05
    # )
    # grid = hv.Layout(edges_dsg + edges_dsg + edges_dsg).cols(3)
    # grid.opts(sublabel_format="")
    # fig = dplot.dsg_to_fig(grid)
    # fig.set_size_inches(0.7 * FIG_WIDTH, 0.25 * FIG_WIDTH)
    # subplots = fig.axes
    # cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])

    fig, subplots = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(0.615 * FIG_WIDTH, 0.235 * FIG_WIDTH),
        width_ratios=(1, 1, 1, 0.05),
    )
    cbar_ax = subplots[-1]

    for seq, axes in zip(seqs, subplots):
        nodes_color = "Jenga_{}".format(seq)
        max_cov = nodes[nodes_color].max()
        nodes[nodes_color] = nodes[nodes_color] / max_cov
        unique_covs = nodes[nodes_color].unique()

        mplot.plot_visualization(
            axes,
            nodes,
            edges_df=edges,
            edges_alpha=0.005,
            nodes_vmax=1,
            nodes_vmin=0,
            nodes_size=1.5,
            nodes_cbar=True,
            nodes_cbar_axes=cbar_ax,
            nodes_cmap_label="K($x_1$, $x_2$)",
            nodes_color=nodes_color,
            nodes_cmap="viridis",
            sort_by=nodes_color,
            sort_ascending=True,
        )

        axes.set(xlabel="", ylabel="")
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

    cbar_ax.set_ylabel("K($x_1$, $x_2$)", fontsize=8)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=8)
    subplots[0].set_ylabel("Diffusion axis 2", fontsize=8)
    subplots[1].set_xlabel("Diffusion axis 1", fontsize=8)

    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/gb1.visualization.kernels_main.png", dpi=300)
