#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpmap.plot.mpl as mplot

from os.path import join
from matplotlib.gridspec import GridSpec
from gpmap.space import SequenceSpace
from scripts.settings import RESULTSDIR, DATADIR, YEAST
from scripts.figures.plot_utils import (
    FIG_WIDTH,
    plot_cv_curve,
    savefig,
    plot_panel_label,
)


def color_by_chr(axes, chr_x, ylim):
    for i in range(0, chr_x.shape[0], 2):
        left, right = chr_x.iloc[i], chr_x.iloc[i + 1]
        axes.fill_between(
            x=(left, right),
            y1=ylim[0],
            y2=ylim[1],
            color="grey",
            alpha=0.15,
            lw=0,
        )


def get_chr_label_and_pos(chr_x):
    chr_pos = []
    prev_pos = 0
    for pos in chr_x:
        chr_pos.append((pos + prev_pos) / 2)
        prev_pos = pos
    chr_labels = ["chr{}".format(i + 1) for i in range(len(chr_pos))]
    return (chr_labels, chr_pos)


def add_gene_labels(axes, data, scale=1):
    genes = ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]
    labels = data.iloc[np.isin(data["gene"], genes), :]

    dxs = [-1, -1, 0.8, 0., 0.5, -1, -0.5, 1, 1.5, -1]
    dys = [0, 1, 1.2, 0, 1, 0, 2, 2, 1, 0]

    for x, y, label, dx, dy in zip(
        labels["x"], labels["decay_rate"], labels["gene"], dxs, dys
    ):
        xtext = x + 2e5 * dx
        ytext = y + (1 + dy) * scale
        if dx == 0:
            ha = 'center'
        else:
            ha = "left" if xtext > x else "right"
            
        if label == 'HAL9':
            ha = 'center'
            
        x = 0.9 * x + 0.1 * xtext
        y = 0.9 * y + 0.1 * ytext

        axes.annotate(
            label,
            xy=(x, y),
            ha=ha,
            xytext=(xtext, ytext),
            fontsize=6,
            arrowprops=dict(
                facecolor="black",
                width=0.25,
                lw=0.3,
                headwidth=2,
                headlength=2,
            ),
        )


def plot_mut_effs(axes, data, bc, color, label):
    axes.errorbar(
        data["x"],
        data["coef_{}".format(bc)],
        yerr=data["stderr_{}".format(bc)],
        color=color,
        marker="o",
        ms=2,
        lw=0,
        elinewidth=1,
        capsize=1,
        label=label,
    )


def add_mut_labels(axes, data):
    genes = ["MKT1", "HAL9", "PHO84", "HAP1"]
    labels = data.iloc[np.isin(data["gene"], genes), :]
    dxs = 0.015 * np.array([0.0, 0.01, 0.0, 0.0, 0.5])
    dys = 0.015 * np.array([-1.25, -0.9, -1.25, 2.5, -1])

    for x, y, label, dx, dy in zip(
        labels["coef_RM_ena1RM"],
        labels["coef_BY_ena1BY"],
        labels["gene"],
        dxs,
        dys,
    ):
        xtext = x + dx
        ytext = y + dy
        
        if dx == 0:
            ha = 'center'
        else:
            ha = "left" if xtext > x else "right"
        va = 'top' if dy < 0 else 'bottom'
            
        axes.text(xtext, ytext, label, fontsize=6, ha=ha, va=va)


def plot_mut_eff_comparison(axes, results, background="RM"):
    axes.errorbar(
        results["coef_{}_ena1RM".format(background)],
        results["coef_{}_ena1BY".format(background)],
        xerr=results["stderr_{}_ena1RM".format(background)],
        yerr=results["stderr_{}_ena1BY".format(background)],
        alpha=0.3,
        color="grey",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=0.8,
        capsize=1,
    )

    results = results.loc[
        np.isin(results["gene"], ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]),
        :,
    ]
    axes.errorbar(
        results["coef_{}_ena1RM".format(background)],
        results["coef_{}_ena1BY".format(background)],
        xerr=results["stderr_{}_ena1RM".format(background)],
        yerr=results["stderr_{}_ena1BY".format(background)],
        alpha=1,
        color="black",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=0.8,
        capsize=0.8,
        zorder=10,
    )
    add_mut_labels(axes, results)
    axes.axline(
        (0, 0.0), (0.1, 0.1), linestyle="--", lw=0.5, c="grey", alpha=0.5
    )
    axes.axvline(0, linestyle="--", lw=0.5, c="grey", alpha=0.5)
    axes.axhline(0, linestyle="--", lw=0.5, c="grey", alpha=0.5)

    lims = (-0.1, 0.1)
    ticks = [-0.1, -0.05, 0, 0.05, 0.1]

    axes.set(
        ylabel="Fitness effects (BY-RM)\nin ENA1-BY",
        xlabel="Fitness effects (BY-RM)\n in ENA1-RM",
        aspect="equal",
        xlim=lims,
        ylim=lims,
        xticks=ticks,
        yticks=ticks,
    )
    axes.text(
        0.05,
        0.95,
        "{} background".format(background),
        transform=axes.transAxes,
        fontsize=8,
        ha="left",
        va="top",
    )


def plot_landscape(axes, landscape, show_edges=True):
    labels = {"A": "ENA1-RM", "B": "ENA1-BY"}
    palette = {"ENA1-BY": "grey", "ENA1-RM": "black"}

    # Create sequence space and edges
    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    edges = space.get_edges_df() if show_edges else None
    landscape["ena1"] = [labels[x] for x in landscape["ena1"]]

    # Plot visualization
    mplot.plot_visualization(
        axes,
        landscape,
        x="d",
        y="coef",
        nodes_size=3,
        edges_df=edges,
        nodes_color="ena1",
        nodes_palette=palette,
        nodes_alpha=0.1,
        edges_color="lightgrey",
        rasterized=True,
    )
    axes.set_xlabel("Hamming distance to best ENA1-RM", fontsize=8)
    axes.set_ylabel("Fitness", fontsize=8)


if __name__ == "__main__":
    dataset = YEAST
    ena1_pos = 12

    print("Loading R2 data")
    fpath = join(RESULTSDIR, "{}.cv_curves.csv".format(dataset))
    r2 = pd.read_csv(fpath, index_col=0)
    r2 = r2.loc[r2["model"] != "Variance Component", :]

    print("Loading measurements")
    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath)
    data["ena1"] = [x[ena1_pos] for x in data["seq"]]
    data["nA"] = [x.count("A") for x in data["seq"]]
    wt = ["A" * 83, "B" * 83]

    print("Loading reconstructed landscape")
    fpath = join(RESULTSDIR, "{}.reconstruction.csv".format(dataset))
    landscape = pd.read_csv(fpath, index_col=0)

    print("Loading model estimates")
    fpath = join(RESULTSDIR, "{}_results.csv".format(dataset))
    results = pd.read_csv(fpath)
    
    print('Loading chromosome coordinates')
    fpath = join(DATADIR, 'raw', "chr_sizes.csv")
    chr_sizes = pd.read_csv(fpath, index_col=0)["size"]
    chr_x = np.cumsum(chr_sizes)

    fig = plt.figure(figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH))
    gs = GridSpec(2, 3)

    print("Plotting R2 curves")
    axes = fig.add_subplot(gs[0, 0])
    plot_cv_curve(axes, r2, metric="r2")
    axes.legend(loc=4)
    plot_panel_label(axes, -0.45, 1.05, "A")

    print("Plotting site-specific decay factors")
    axes = fig.add_subplot(gs[0, 1:])
    axes.scatter(results["x"], results["decay_rate"], c="black", s=5, lw=0)
    ylim = (-0.5, 60)
    color_by_chr(axes, chr_x, ylim)
    chr_labels, chr_pos = get_chr_label_and_pos(chr_x)
    axes.set(
        ylabel="Decay factor (%)",
        ylim=ylim,
        xlim=(0, chr_x.max()),
        xlabel="Chromomsome position (bp)",
        xticks=chr_pos,
    )
    add_gene_labels(axes, results, scale=5)
    axes.set_xticklabels(chr_labels, rotation=45)
    plot_panel_label(axes, -0.15, 1.05, "B")

    print("Plotting fitness distribution on ENA1 alleles")
    axes = fig.add_subplot(gs[1, 0])
    bins = np.linspace(data["y"].min(), data["y"].max(), 50)
    kwargs = {"alpha": 0.5, "bins": bins}
    idx = data["ena1"] == "A"
    axes.hist(data.loc[~idx, "y"], color="grey", label="ENA1-BY", **kwargs)
    axes.hist(data.loc[idx, "y"], color="black", label="ENA1-RM", **kwargs)
    axes.legend(loc=0)
    axes.set(xlabel=r"Measured fitness", ylabel="# segregants")
    plot_panel_label(axes, -0.35, 1.05, "C")

    print("Plotting reconstructed landscape")
    axes = fig.add_subplot(gs[1, 1])
    plot_landscape(axes, landscape)
    plot_panel_label(axes, -0.25, 1.05, "D")

    print("Plotting fitness effects on ENA1 alleles")
    axes = fig.add_subplot(gs[1, 2])
    results = results.loc[results["gene"] != "ENA1", :]
    plot_mut_eff_comparison(axes, results, background="RM")
    plot_panel_label(axes, -0.6, 1.05, "E")

    fig.subplots_adjust(
        hspace=0.6, wspace=0.55, top=0.95, right=0.95, left=0.1, bottom=0.15
    )
    savefig(fig, "{}.main_figure".format(dataset))
