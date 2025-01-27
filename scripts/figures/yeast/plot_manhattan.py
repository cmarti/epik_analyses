#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def color_by_chr(axes, chr_x, ylim):
    for i in range(0, chr_x.shape[0], 2):
        left, right = chr_x.iloc[i], chr_x.iloc[i + 1]
        axes.fill_between(
            x=(left, right),
            y1=ylim[0],
            y2=ylim[1],
            color="grey",
            alpha=0.2,
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
        axes.annotate(
            label,
            xy=(x, y + 0.1),
            ha=ha,
            xytext=(xtext, ytext),
            fontsize=7,
            arrowprops=dict(
                facecolor="black",
                shrink=1,
                width=0.25,
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


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    dataset = "qtls_li_hq"

    fpath = "results/{}_results.csv".format(dataset)
    data = pd.read_csv(fpath)
    print(data)

    chr_sizes = pd.read_csv("raw/chr_sizes.csv", index_col=0)["size"]
    chr_x = np.cumsum(chr_sizes)

    fig, subplots = plt.subplots(2, 1, figsize=(8, 4.75), sharex=True)

    # Plot decay factors
    axes = subplots[0]
    axes.scatter(data["x"], data["decay_rate"], c="black", s=7.5, lw=0)
    ylim = (-0.5, 60)
    color_by_chr(axes, chr_x, ylim)
    axes.grid(alpha=0.1)
    chr_labels, chr_pos = get_chr_label_and_pos(chr_x)
    axes.set(
        ylabel="Decay factor (%)",
        ylim=ylim,
        xlim=(0, chr_x.max()),
        xlabel="",
        xticks=[],
    )
    add_gene_labels(axes, data, scale=5)

    # Mutational effects
    axes = subplots[1]
    plot_mut_effs(
        axes, data, bc="BY_ena1BY", color="grey", label="BY background"
    )
    plot_mut_effs(
        axes, data, bc="RM_ena1RM", color="black", label="RM background"
    )

    axes.legend(loc=1)
    ylim = (-0.075, 0.25)
    color_by_chr(axes, chr_x, ylim)
    axes.grid(alpha=0.1)
    chr_labels, chr_pos = get_chr_label_and_pos(chr_x)
    axes.set(
        ylabel="Fitness effects (BY - RM)",
        ylim=ylim,
        xlim=(0, chr_x.max()),
        xlabel="Chromomsome position (bp)",
        xticks=chr_pos,
    )

    axes.set_xticklabels(chr_labels, rotation=45)

    fig.tight_layout()
    fig.savefig("figures/{}.manhattan.png".format(dataset), dpi=300)
    fig.savefig("figures/{}.manhattan.svg".format(dataset), dpi=300)
