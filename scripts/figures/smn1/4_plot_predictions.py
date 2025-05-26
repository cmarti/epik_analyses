#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


from scripts.figures.plot_utils import FIG_WIDTH


def plot_scatter(x, y, axes, vmin=0, vmax=3):
    r2 = pearsonr(x, y)[0] ** 2
    rmse = np.sqrt(np.mean((x - y) ** 2))

    lims = min(x.min(), y.min()), max(x.max(), y.max())
    bins = np.linspace(lims[0], lims[1], 100)
    diff = lims[1] - lims[0]
    lims = (lims[0] - 0.05 * diff, lims[1] + 0.05 * diff)

    H, xbins, ybins = np.histogram2d(x=x, y=y, bins=bins)
    im = axes.imshow(
        np.log10(H.T[::-1, :]),
        cmap="viridis",
        extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
        vmin=vmin,
        vmax=vmax,
    )
    axes.plot(lims, lims, lw=0.5, linestyle="--", c="black")
    axes.text(
        0.95,
        0.05,
        "$R^2$={:.2f}\nRMSE={:.2f}".format(r2, rmse),
        transform=axes.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ticks = [0, 50, 100, 150]
    axes.set(
        xlabel=r"Predicted PSI (%)",
        ylabel=r"Observed PSI (%)",
        xlim=lims,
        ylim=lims,
        aspect="equal",
        xticks=ticks,
        yticks=ticks,
    )
    return im


if __name__ == "__main__":
    dataset = "smn1"
    kernel = "Jenga"
    i = 60

    data = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0)
    fpath = "output_new/{}.{}.{}.test_pred.csv".format(dataset, i, kernel)
    pred = pd.read_csv(fpath, index_col=0).join(data).dropna()

    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.44, FIG_WIDTH * 0.4))
    x, y = pred["coef"].values, pred["y"].values
    im = plot_scatter(x, y, axes)
    fig.colorbar(im, label="# test sequences", shrink=0.5)
    axes = fig.axes[-1]
    axes.set_yticks([0, 1, 2, 3])
    axes.set_yticklabels(["10$^0$", "10$^1$", "10$^2$", "10$^3$"])

    fig.tight_layout()
    fig.savefig("figures/{}.scatter.png".format(dataset), dpi=300)
    fig.savefig("figures/{}.scatter.svg".format(dataset), dpi=300)
