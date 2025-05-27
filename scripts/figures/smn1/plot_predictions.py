#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
from scripts.settings import SMN1, DATADIR, PARAMSDIR
from scripts.figures.plot_utils import FIG_WIDTH, savefig, plot_2D_hist


if __name__ == "__main__":
    dataset = SMN1
    kernel = "Jenga"
    i = 60

    print("Loading data for {} with {} kernel".format(dataset, kernel))
    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)

    fname = "{}.{}.{}.test_pred.csv".format(dataset, i, kernel)
    pred = pd.read_csv(join(PARAMSDIR, fname), index_col=0).join(data).dropna()

    print("Plotting 2D histogram of predictions")
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.44, FIG_WIDTH * 0.4))
    x, y = pred["coef"].values, pred["y"].values
    im = plot_2D_hist(x, y, axes)
    fig.colorbar(im, label="# test sequences", shrink=0.5)
    axes = fig.axes[-1]
    axes.set_yticks([0, 1, 2, 3])
    axes.set_yticklabels(["10$^0$", "10$^1$", "10$^2$", "10$^3$"])

    fig.tight_layout()
    savefig(fig, "{}.{}.scatter".format(dataset, kernel))
