#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_gene_labels(axes, data):
    genes = ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]
    labels = data.iloc[np.isin(data["gene"], genes), :]

    dxs = 0.025 * np.array([1, 1, 1, 1, 1])
    dys = 0.025 * np.array([-1, -1, -1, -1, -1])

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
        axes.annotate(
            label,
            xy=(x, y),
            ha=ha,
            xytext=(xtext, ytext),
            fontsize=9,
            # arrowprops=dict(
            #     facecolor="black",
            #     shrink=1,
            #     width=0.25,
            #     headwidth=2,
            #     headlength=2,
            # ),
        )


if __name__ == "__main__":
    dataset = "qtls_li_hq"

    fpath = "results/{}_results.csv".format(dataset)
    data = pd.read_csv(fpath)
    print(data)

    fig, axes = plt.subplots(1, 1, figsize=(3.5, 2.5))

    axes.errorbar(
        data["coef_RM_ena1RM"],
        data["coef_BY_ena1BY"],
        xerr=data["stderr_RM_ena1RM"],
        yerr=data["stderr_BY_ena1BY"],
        alpha=0.3,
        color="grey",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=1,
        capsize=1,
    )

    data = data.loc[
        np.isin(data["gene"], ["ENA1", "MKT1", "HAL9", "PHO84", "HAP1"]),
        :,
    ]
    axes.errorbar(
        data["coef_RM_ena1RM"],
        data["coef_BY_ena1BY"],
        xerr=data["stderr_RM_ena1RM"],
        yerr=data["stderr_BY_ena1BY"],
        alpha=1,
        color="black",
        marker="o",
        ms=2,
        lw=0,
        elinewidth=1,
        capsize=1,
        zorder=10,
    )
    add_gene_labels(axes, data)

    axes.grid(alpha=0.1)
    axes.axline((0, 0.0), (0.1, 0.1), linestyle="--", lw=0.75, c="grey")
    axes.axvline(0, linestyle="--", lw=0.75, c="grey")
    axes.axhline(0, linestyle="--", lw=0.75, c="grey")

    lims = (-0.1, 0.25)
    ticks = [-0.1, 0, 0.1, 0.2]

    axes.set(
        ylabel="Fitness effects in BY",
        xlabel="Fitness effects in RM",
        aspect="equal",
        xlim=lims,
        ylim=lims,
        xticks=ticks,
        yticks=ticks,
    )

    fig.tight_layout()
    fig.savefig("figures/{}.mut_effs_scatter.png".format(dataset), dpi=300)
    fig.savefig("figures/{}.mut_effs_scatter.svg".format(dataset), dpi=300)
