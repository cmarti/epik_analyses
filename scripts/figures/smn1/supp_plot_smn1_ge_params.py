#!/usr/bin/env python
import numpy as np
import mavenn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from matplotlib import colormaps
from scripts.settings import SMN1, PARAMSDIR, DATADIR
from scripts.figures.plot_utils import FIG_WIDTH, highlight_seq_heatmap, plot_2D_hist, savefig


def read_theta():
    dataset = "smn1"
    positions = ["-3", "-2", "-1", "+2", "+3", "+4", "+5", "+6"]
    fpath = join(PARAMSDIR, "{}.global_epistasis.model".format(dataset))
    model = mavenn.load(fpath)
    theta = model.get_theta()
    theta_lc = theta["logomaker_df"]
    theta_lc.index = positions
    theta_lc_max = theta_lc.max(axis=1).values.reshape((theta_lc.shape[0], 1))
    theta_lc = theta_lc - theta_lc_max
    sd = np.abs(
        np.nanmean(theta_lc.values)
        * theta_lc.shape[1]
        / (theta_lc.shape[1] - 1)
    )
    theta_lc = theta_lc / sd

    seq0 = "".join([theta_lc.columns[i] for i in np.where(theta_lc == 0.0)[1]])
    phi_max = model.x_to_phi(seq0)

    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)
    data["phi"] = model.x_to_phi(data.index.values)
    delta = 0.025 * (phi_max - data["phi"].min())

    phi = np.linspace(data["phi"].min() - delta, phi_max + delta, 101)
    yhat = model.phi_to_yhat(phi)
    phi = (phi - phi_max) / sd
    pred = pd.DataFrame({"phi": phi, "yhat": yhat})
    data["phi"] = (data["phi"] - phi_max) / sd
    return (theta_lc, pred, data.dropna())


def plot_theta_heatmap(axes, theta):
    cmap = colormaps["binary"]
    axes.set_facecolor(cmap(0.1))
    sns.heatmap(
        theta.T,
        ax=axes,
        cmap="Blues_r",
        vmax=0,
        cbar_kws={"label": r"$\Delta\phi$", "shrink": 0.55},
    )
    axes.set(xlabel="Position", ylabel="Allele", aspect="equal")
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
    highlight_seq_heatmap(axes, theta, dataset)
    sns.despine(ax=axes, right=False, top=False)


def plot_nonlinearity(axes, pred, data):
    H, xbins, ybins = np.histogram2d(
        x=data["phi"].values,
        y=data["y"].values,
        bins=60,
    )
    dy = ybins[-1] - ybins[0]
    aspect = (xbins[-1] - xbins[0]) / dy
    with np.errstate(divide='ignore'):
        im = axes.imshow(
            np.log(H.T[::-1, :]),
            cmap="viridis",
            extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
            aspect=aspect,
        )
    plt.colorbar(im, shrink=0.7, label='# sequences')
    axes.plot(pred["phi"], pred["yhat"], c="black", lw=1)
    axes.set(
        xlim=(pred["phi"].min(), pred["phi"].max()),
        xlabel=r"Latent phenotype $\phi$",
        ylabel=r"Observed PSI",
    )
    


if __name__ == "__main__":
    dataset = SMN1
    kernel = "VC"
    i = 60
    
    
    print("Loading {} global epistasis parameters".format(dataset))
    theta, pred, data = read_theta()

    print('Plotting {} global epistasis fit'.format(dataset))
    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH, 0.25 * FIG_WIDTH),
        width_ratios=[5, 6, 5],
    )
    plot_nonlinearity(subplots[0], pred, data)
    subplots[0].set_title("Global epistasis")
    fig.axes[-1].set_yticks([0, np.log(10), np.log(100), np.log(1000)])
    fig.axes[-1].set_yticklabels([r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$'])
    
    print('Plotting {} global epistasis parameters'.format(dataset))
    plot_theta_heatmap(subplots[1], theta)
    sns.despine(ax=fig.axes[-1], right=False, top=False)
    fig.axes[-1].set_yticks([-2, -1, 0])
    
    print("Loading VC regression predictions")
    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)
    fname = "{}.{}.{}.test_pred.csv".format(dataset, i, kernel)
    pred = pd.read_csv(join(PARAMSDIR, fname), index_col=0).join(data).dropna()
    
    print('Plotting VC regression predictions')
    axes = subplots[2]
    x, y = pred["coef"].values, pred["y"].values
    im = plot_2D_hist(x, y, axes)
    axes.set_title('Variance component regression')
    fig.colorbar(im, label="# test sequences", shrink=0.7)
    axes = fig.axes[-1]
    axes.set_yticks([0, 1, 2, 3])
    axes.set_yticklabels(["10$^0$", "10$^1$", "10$^2$", "10$^3$"])

    fig.tight_layout()
    savefig(fig, "{}.global_epistasis".format(dataset))
