#!/usr/bin/env python
import numpy as np
import holoviews as hv
import gpmap.plot.mpl as mplot
import gpmap.plot.ds as dplot

from os.path import join
from scripts.utils import load_gb1_visualization
from scripts.figures.plot_utils import FIG_WIDTH
from scripts.settings import FIGDIR, MODELS, GB1_PEAK_SEQS, MODEL_KEYS


if __name__ == "__main__":
    hv.extension("matplotlib")
    labels = [x for x in MODELS if x != 'Global epistasis']
    nrows, ncols = len(GB1_PEAK_SEQS), len(labels)
    nodes, edges = load_gb1_visualization()

    print("Rendering edges for visualization")
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

    print("Plotting kernels on the visualization")
    for label, axes_row in zip(labels, subplots):
        
        print("\tPlotting kernel {}".format(label))
        kernel_label = MODEL_KEYS.get(label, label)
        
        for seq, axes in zip(GB1_PEAK_SEQS, axes_row):
            
            nodes_color = "{}_{}".format(kernel_label, seq)
            max_cov = nodes[nodes_color].max()
            nodes[nodes_color] = nodes[nodes_color] / max_cov
            
            mplot.plot_visualization(
                axes,
                nodes,
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
                rasterized=True,
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
    fpath = join(FIGDIR, "gb1.visualization.kernels.png")
    fig.savefig(fpath, dpi=300)
