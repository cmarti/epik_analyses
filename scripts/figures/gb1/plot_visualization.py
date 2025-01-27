#!/usr/bin/env python
import pandas as pd

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as plot

from gpmap.src.utils import read_edges


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    kernel_visualization = "Jenga"

    # Load kernel function
    kernel = pd.read_csv("output/gb1.kernels_at_peaks.csv", index_col=0)

    # Load visualization
    nodes = pd.read_csv(
        "output/gb1.{}.nodes.csv".format(kernel_visualization), index_col=0
    )
    edges = read_edges("output/gb1.edges.npz")
    nodes = nodes.join(kernel)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(3.25, 2.55),
        sharex=False,
        sharey=False,
        width_ratios=(1, 0.05),
    )

    axes = subplots[0]
    cbar_ax = subplots[-1]
    plot.plot_visualization(
        axes,
        nodes,
        edges_df=edges,
        edges_alpha=0.01,
        nodes_cbar=True,
        nodes_cbar_axes=cbar_ax,
        nodes_cmap_label="log(Enrichment)",
        nodes_color="function",
        nodes_cmap="coolwarm",
        sort_by="3",
        sort_ascending=True,
        fontsize=11,
    )
    axes.set(
        xlabel="Diffusion axis 1", ylabel="Diffusion axis 2", aspect="equal"
    )
    axes.grid(alpha=0.2, lw=0.5)

    fig.tight_layout(w_pad=0.05)
    fig.savefig("figures/gb1.visualization.png", dpi=300)
