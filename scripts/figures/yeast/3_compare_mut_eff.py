#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from scripts.figures.plot_utils import FIG_WIDTH


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


def add_mut_labels(axes, data, x, y):
    genes = ["MKT1", "HAL9", "PHO84", "HAP1"]
    labels = data.iloc[np.isin(data["gene"], genes), :]

    dxs = 0.015 * np.array([-0.5, 0.01, 1.4, 0.5, 1]) * 0
    dys = 0.015 * np.array([0.5, -1.5, -0.75, 1.5, -1]) * 0

    for x, y, label, dx, dy in zip(
        labels["coef_{}".format(x)],
        labels["coef_{}".format(y)],
        labels["gene"],
        dxs,
        dys,
    ):
        xtext = x + dx
        ytext = y + dy
        ha = "left" if xtext > x else "right"
        axes.annotate(
            label,
            xy=(x, y),
            ha=ha,
            xytext=(xtext, ytext),
            fontsize=6,
            # arrowprops=dict(
            #     facecolor="black",
            #     shrink=1,
            #     width=0.25,
            #     headwidth=2,
            #     headlength=2,
            # ),
        )


if __name__ == "__main__":
    bcs = ["RM_ena1RM", "BY_ena1RM", "RM_ena1BY", "BY_ena1BY"]
    labels = [
        "ENA1-RM in RM background",
        "ENA1-RM in BY background",
        "ENA1-BY in RM background",
        "ENA1-BY in BY background",
    ]
    fpath = "results/qtls_li_hq_results.csv"
    data = pd.read_csv(fpath)
    data = data.loc[data["gene"] != "ENA1", :]

    fig, subplots = plt.subplots(
        2, 3, figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH), sharex=True, sharey=True
    )
    subplots = subplots.flatten()

    for (i, j), axes in zip(combinations(np.arange(4), 2), subplots):
        bc1, label1 = bcs[i], labels[i]
        bc2, label2 = bcs[j], labels[j]
        axes.errorbar(
            data["coef_{}".format(bc1)],
            data["coef_{}".format(bc2)],
            xerr=data["stderr_{}".format(bc1)],
            yerr=data["stderr_{}".format(bc2)],
            alpha=0.3,
            color="grey",
            marker="o",
            ms=2,
            lw=0,
            elinewidth=0.8,
            capsize=1,
        )

        df = data.loc[
            np.isin(data["gene"], ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]),
            :,
        ]
        axes.errorbar(
            df["coef_{}".format(bc1)],
            df["coef_{}".format(bc2)],
            xerr=df["stderr_{}".format(bc1)],
            yerr=df["stderr_{}".format(bc2)],
            alpha=1,
            color="black",
            marker="o",
            ms=2,
            lw=0,
            elinewidth=0.8,
            capsize=0.8,
            zorder=10,
        )
        add_mut_labels(axes, data, x=bc1, y=bc2)
        axes.axline(
            (0, 0.0), (0.1, 0.1), linestyle="--", lw=0.5, c="grey", alpha=0.5
        )
        axes.axvline(0, linestyle="--", lw=0.5, c="grey", alpha=0.5)
        axes.axhline(0, linestyle="--", lw=0.5, c="grey", alpha=0.5)

        lims = (-0.1, 0.1)
        ticks = [-0.1, -0.05, 0, 0.05, 0.1]

        axes.set(
            xlabel="{}".format(label1),
            ylabel="{}".format(label2),
            xlim=lims,
            ylim=lims,
            aspect="equal",
            xticks=ticks,
            yticks=ticks,
        )

    fig.supxlabel("Fitness effects (BY-RM)", fontsize=9, x=0.55, ha="center")
    fig.supylabel("Fitness effects (BY-RM)", fontsize=9, y=0.55, va="center")

    fig.tight_layout()
    fig.savefig("figures/yeast_supp_fig.png", dpi=300)
    fig.savefig("figures/yeast_supp_fig.svg", dpi=300)
