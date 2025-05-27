#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpmap.plot.mpl as mplot

from matplotlib.gridspec import GridSpec
from gpmap.space import SequenceSpace
from scripts.figures.plot_utils import FIG_WIDTH, plot_cv_curve


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

    dxs = [-1, -1, 1, 1, 1, -1, -0.5, 1, 1.5, -1]
    dys = [0, 1, 0.7, 0, 1, 0, 2, 2, 1, 0]

    for x, y, label, dx, dy in zip(
        labels["x"], labels["decay_rate"], labels["gene"], dxs, dys
    ):
        xtext = x + 2e5 * dx
        ytext = y + (1 + dy) * scale
        ha = "left" if xtext > x else "right"
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
                # shrink=0.01,
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
    print(labels["gene"])
    dxs = 0.015 * np.array([-0.5, 0.1, 0.2, 0.1, 0.5])
    dys = 0.015 * np.array([0.5, -1.5, -0.75, 1.5, -1])

    for x, y, label, dx, dy in zip(
        labels["coef_RM_ena1RM"],
        labels["coef_BY_ena1BY"],
        labels["gene"],
        dxs,
        dys,
    ):
        xtext = x + dx
        ytext = y + dy
        ha = "left" if xtext > x else "right"
        print(label, xtext, ytext)
        axes.text(xtext, ytext, label, fontsize=6, ha=ha)
        # axes.annotate(
        #     label,
        #     xy=(x, y),
        #     ha=ha,
        #     xytext=(xtext, ytext),
        #     fontsize=6,
        #     arrowprops=dict(
        #         facecolor="black",
        #         shrink=1,
        #         width=0.25,
        #         headwidth=2,
        #         headlength=2,
        #     ),
        # )


if __name__ == "__main__":
    dataset = "qtls_li_hq"
    fig = plt.figure(figsize=(0.65 * FIG_WIDTH, 0.85 * FIG_WIDTH))
    gs = GridSpec(3, 2)
    background = "RM"

    print("Plotting R2 curves")
    axes = fig.add_subplot(gs[0, 0])
    fpath = "results/{}.cv_curves.csv".format(dataset)
    data = pd.read_csv(fpath, index_col=0)
    data = data.loc[data["model"] != "Variance Component", :]
    plot_cv_curve(axes, data, metric="r2")
    axes.legend(loc=4)
    axes.text(
        -0.45,
        1.05,
        "A",
        transform=axes.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    print("Plotting fitness distribution on ENA1 alleles")
    axes = fig.add_subplot(gs[0, 1])
    ena1_pos = 12
    data = pd.read_csv("datasets/qtls_li_hq.csv")
    data["ena1"] = [x[ena1_pos] for x in data["seq"]]
    data["nA"] = [x.count("A") for x in data["seq"]]
    wt = ["A" * 83, "B" * 83]
    bins = np.linspace(data["y"].min(), data["y"].max(), 50)
    idx = data["ena1"] == "A"
    axes.hist(
        data.loc[~idx, "y"], color="grey", label="ENA1-BY", alpha=0.5, bins=bins
    )
    axes.hist(
        data.loc[idx, "y"], color="black", label="ENA1-RM", alpha=0.5, bins=bins
    )
    axes.legend(loc=0)
    axes.set(xlabel=r"Measured fitness", ylabel="# segregants")
    axes.text(
        -0.35,
        1.05,
        "C",
        transform=axes.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    print("Plotting fitness effects on ENA1 alleles")
    axes = fig.add_subplot(gs[2, 1])
    fpath = "results/{}_results.csv".format(dataset)
    data = pd.read_csv(fpath)
    data = data.loc[data["gene"] != "ENA1", :]

    print(
        data.sort_values("upper_ci_BY_ena1RM")[
            [
                "gene",
                "coef_BY_ena1RM",
                "upper_ci_BY_ena1RM",
                "lower_ci_BY_ena1RM",
            ]
        ]
    )
    axes.errorbar(
        data["coef_{}_ena1RM".format(background)],
        data["coef_{}_ena1BY".format(background)],
        xerr=data["stderr_{}_ena1RM".format(background)],
        yerr=data["stderr_{}_ena1BY".format(background)],
        alpha=0.3,
        color="grey",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=0.8,
        capsize=1,
    )

    data = data.loc[
        np.isin(data["gene"], ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]),
        :,
    ]
    axes.errorbar(
        data["coef_{}_ena1RM".format(background)],
        data["coef_{}_ena1BY".format(background)],
        xerr=data["stderr_{}_ena1RM".format(background)],
        yerr=data["stderr_{}_ena1BY".format(background)],
        alpha=1,
        color="black",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=0.8,
        capsize=0.8,
        zorder=10,
    )
    add_mut_labels(axes, data)
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
    axes.text(
        -0.6,
        1.05,
        "E",
        transform=axes.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    print("Plotting site-specific decay factors")
    axes = fig.add_subplot(gs[1, :])
    fpath = "results/{}_results.csv".format(dataset)
    data = pd.read_csv(fpath)
    chr_sizes = pd.read_csv("raw/chr_sizes.csv", index_col=0)["size"]
    chr_x = np.cumsum(chr_sizes)
    print(data.loc[data["gene"] == "ENA1", ["x", "decay_rate"]])
    # exit()
    axes.scatter(data["x"], data["decay_rate"], c="black", s=5, lw=0)
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
    add_gene_labels(axes, data, scale=5)
    axes.set_xticklabels(chr_labels, rotation=45)
    axes.text(
        -0.15,
        1.05,
        "B",
        transform=axes.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    print("Plotting reconstructed landscape")
    axes = fig.add_subplot(gs[2, 0])
    fpath = "output_new/qtls_li_hq.Connectedness.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)
    loci = np.load("datasets/qtls_li_hq.selected_loci.npy", allow_pickle=True)
    loci = np.append(["BC"], loci)
    with open("datasets/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]
    landscape.index = seqs
    ena1_idx = list(loci).index("ENA1")
    landscape["ena1"] = [seq[ena1_idx] for seq in landscape.index]
    best_ena1_rm = landscape.loc[landscape["ena1"] == "A", "coef"].idxmax()
    landscape["d"] = [
        sum(c1 != c2 for c1, c2 in zip(seq, best_ena1_rm))
        for seq in landscape.index
    ]
    landscape["d"] += np.random.normal(0, 0.075, size=landscape.shape[0])

    # Create sequence space and edges
    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    edges = space.get_edges_df()
    labels = {"A": "ENA1-RM", "B": "ENA1-BY"}
    landscape["ena1"] = [labels[x] for x in landscape["ena1"]]
    palette = {"ENA1-BY": "grey", "ENA1-RM": "black"}

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
    axes.text(
        -0.25,
        1.05,
        "D",
        transform=axes.transAxes,
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.savefig("figures/{}.main_figure.png".format(dataset), dpi=300)
    fig.savefig("figures/{}.main_figure.svg".format(dataset), dpi=300)
    fig.savefig("figure_yeast.png", dpi=300)
    fig.savefig("figure_yeast.svg", dpi=300)
