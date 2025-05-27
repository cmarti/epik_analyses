#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from os.path import join
from scripts.settings import AAV, RESULTSDIR, DATADIR
from scripts.figures.plot_utils import FIG_WIDTH, savefig


def draw_guide_lines(axes):
    kwargs = {"lw": 0.5, "color": "grey", "alpha": 0.5, "linestyle": "--"}
    axes.axline((0, 0), (1, 1), **kwargs)
    axes.axvline(0, **kwargs)
    axes.axhline(0, **kwargs)


def plot_mut_effs_scatter(axes, mut_eff, bc1, bc2):
    axes.errorbar(
        mut_eff["coef_{}".format(bc1)],
        mut_eff["coef_{}".format(bc2)],
        xerr=2 * mut_eff["stderr_{}".format(bc1)],
        yerr=2 * mut_eff["stderr_{}".format(bc2)],
        fmt="",
        color="grey",
        alpha=0.25,
        elinewidth=0.5,
        markersize=1.5,
        lw=0,
    )
    axes.scatter(
        mut_eff["coef_{}".format(bc1)],
        mut_eff["coef_{}".format(bc2)],
        color="black",
        alpha=0.75,
        s=2,
        lw=0,
        zorder=10,
    )
    draw_guide_lines(axes)

    lims = (-11, 5)
    ticks = list(range(-10, 5, 2))
    axes.set(
        xlabel="Mutational effect in {}".format(bc1),
        ylabel="Mutational effect in {}".format(bc2),
        aspect="equal",
        xlim=lims,
        ylim=lims,
        xticks=ticks,
        yticks=ticks,
    )


def get_background_phenotypes(data, site):
    ref = 561
    label = str(site)
    data[label] = [x[site - ref] for x in data.index]
    data["background"] = [
        x[: site - ref] + x[site - ref + 1 :] for x in data.index
    ]
    df = pd.pivot_table(data, index="background", columns=label, values="y")
    df["n"] = np.isnan(df).sum(1)
    df = df.loc[df["n"] < 19, :].dropna(subset=["N"])
    return df.drop("n", axis=1).reset_index()


if __name__ == "__main__":
    dataset = AAV

    print("Loading estimated mutational effects in {}".format(dataset))
    fpath = join(RESULTSDIR, "{}.mutational_effects.csv".format(dataset))
    mut_effs = pd.read_csv(fpath, index_col=0)

    print("Loading {} data".format(dataset))
    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)
    lims = data["y"].min(), data["y"].max()

    fig, subplots = plt.subplots(2, 2, figsize=(0.66 * FIG_WIDTH, 0.66 * FIG_WIDTH))

    axes = subplots[0][0]
    plot_mut_effs_scatter(axes, mut_effs, "WT", "N569Q")

    axes = subplots[0][1]
    df = get_background_phenotypes(data, 569)
    df = pd.melt(df, id_vars=["N", "background"]).dropna()
    axes.scatter(df["N"], df["value"], color="black", alpha=0.75, s=3, lw=0)

    draw_guide_lines(axes)
    ticks = [-7.5, -5, -2.5, 0, 2.5, 5, 7.5]
    axes.set(
        xlabel="569-Asn DMS score",
        ylabel="569-Other DMS score",
        xlim=lims,
        ylim=lims,
        aspect="equal",
        xticks=ticks,
        yticks=ticks,
    )
    
    axes = subplots[1][0]
    plot_mut_effs_scatter(axes, mut_effs, "WT", "Y576F")
    
    axes = subplots[1][1]
    plot_mut_effs_scatter(axes, mut_effs, "WT", "Y576C")

    fig.tight_layout()
    savefig(fig, "aav_mut_effs_supp1", dpi=300)
