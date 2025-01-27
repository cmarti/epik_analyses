#!/usr/bin/env python
import pandas as pd

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as plot

from gpmap.src.utils import read_edges


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"

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
    kernel = pd.read_csv("output/gb1.kernels_at_peaks.csv", index_col=0)

    # Load visualization
    nodes = pd.read_csv(
        "output/gb1.{}.nodes.csv".format(kernel_visualization), index_col=0
    )
    edges = read_edges("output/gb1.edges.npz")
    nodes = nodes.join(kernel)

    fig, subplots = plt.subplots(
        ncols,
        nrows,
        figsize=(3.5 * nrows, 2.5 * ncols),
        sharex=True,
        sharey=True,
    )

    for label, axes_row in zip(labels, subplots):
        print("PLotting kernel {}".format(label))
        for seq, axes in zip(seqs, axes_row):
            nodes_color = "{}_{}".format(label, seq)
            max_cov = nodes[nodes_color].max()
            nodes[nodes_color] = nodes[nodes_color] / max_cov
            unique_covs = nodes[nodes_color].unique()
            print("\t", seq, unique_covs.shape, max_cov)
            if unique_covs.shape[0] < 20:
                print(unique_covs)
            plot.plot_visualization(
                axes,
                nodes,
                # edges_df=edges,
                edges_alpha=0.05,
                nodes_vmax=1,
                nodes_vmin=0,
                nodes_cbar=True,
                nodes_cmap_label="K({}, x)".format(seq),
                nodes_color=nodes_color,
                nodes_cmap="viridis",
                sort_by=nodes_color,
                sort_ascending=True,
                fontsize=11,
            )
            axes.set(xlabel="", ylabel="")
            axes.text(
                0.98,
                0.98,
                label,
                transform=axes.transAxes,
                ha="right",
                va="top",
            )
    for axes in subplots[-1]:
        axes.set(xlabel="Diffusion axis 1")
    for axes in subplots[:, 0]:
        axes.set(ylabel="Diffusion axis 2")

    fig.tight_layout()
    fig.savefig("figures/gb1.visualization.kernels.png", dpi=300)
