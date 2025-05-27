#!/usr/bin/env python
import matplotlib.pyplot as plt
import gpmap.plot.mpl as mplot

from scripts.settings import GB1_PEAK_SEQS
from scripts.utils import load_gb1_visualization
from scripts.figures.plot_utils import FIG_WIDTH, savefig


def plot_visualization_prior(axes, nodes, edges, seq):
    nodes_color = "Jenga_{}".format(seq)
    max_cov = nodes[nodes_color].max()
    nodes[nodes_color] = nodes[nodes_color] / max_cov

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
    if seq != GB1_PEAK_SEQS[0]:
        axes.set(yticklabels=[])


if __name__ == "__main__":
    nrows, ncols = 1, len(GB1_PEAK_SEQS)
    nodes, edges = load_gb1_visualization()

    print("Plotting prior correlations on the visualization")
    fig, subplots = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(0.615 * FIG_WIDTH, 0.235 * FIG_WIDTH),
        width_ratios=(1, 1, 1, 0.05),
    )
    cbar_ax = subplots[-1]

    for seq, axes in zip(GB1_PEAK_SEQS, subplots):
        print("\tPlotting prior correlations with {}".format(seq))
        plot_visualization_prior(axes, nodes, edges, seq)

    cbar_ax.set_ylabel("K($x_1$, $x_2$)", fontsize=8)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=8)
    subplots[0].set_ylabel("Diffusion axis 2", fontsize=8)
    subplots[1].set_xlabel("Diffusion axis 1", fontsize=8)

    fig.tight_layout(w_pad=0.2)
    savefig(fig, "gb1.visualization.kernels_main")
