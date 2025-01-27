#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import logsumexp

from scripts.figures.utils import load_decay_rates, load_params
from scripts.figures.plot_utils import FIG_WIDTH
from scripts.figures.settings import ALPHABET, POSITIONS


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    dataset = "gb1"
    alphabet = ALPHABET[dataset]
    positions = POSITIONS[dataset]

    params = load_params(dataset=dataset, kernel="GeneralProduct", id=34)
    log_var0 = params["covar_module.module.log_var"][0].item()
    with open("gb1.Jenga.log_var0.txt", "w") as fhand:
        fhand.write("{}".format(log_var0))

    general_product = load_decay_rates(
        dataset=dataset, kernel="GeneralProduct", id=34
    )

    theta = []
    for pos in positions:
        C = 1 - general_product[pos]

        def params_to_corr(params):
            log_rho, log_p = params[0], params[1:]
            rho = np.exp(log_rho)
            p = np.exp(log_p - logsumexp(log_p))
            eta = (1 - p) / p
            fs = np.sqrt(1 + eta * rho)
            corr = (1 - rho.reshape((1, 1))) / (
                np.expand_dims(fs, 0) * np.expand_dims(fs, 1)
            )
            np.fill_diagonal(corr, 1)
            return corr

        def loss(params):
            corr = params_to_corr(params)
            return np.square(C.values - corr).sum()

        params0 = np.zeros(21)
        res = minimize(loss, x0=params0)
        theta.append(res.x)

    theta = pd.DataFrame(np.vstack(theta), index=positions)
    theta.to_csv("gb1.Jenga.theta0.csv")
    print(theta)

    exit()

    # Mutation specific decay factors
    figsize = (0.325 * FIG_WIDTH, 0.65 * FIG_WIDTH)
    fig, subplots = plt.subplots(2, 1, figsize=figsize, sharex=True)

    cbar_axes = fig.add_axes([0.65, 0.3, 0.04, 0.4])
    # fig.subplots_adjust(right=0.60, left=0.25)

    axes = subplots[0]
    pos = "41"
    df = general_product[pos] * 100
    print(df.max())
    sns.heatmap(
        df,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar=False,
    )
    ticks = np.arange(len(alphabet)) + 0.5
    axes.set(
        title="Position {}".format(pos),
        ylabel="Allele 2",
        xlabel="",
        xticks=ticks,
        yticks=ticks,
        xticklabels=[],
        aspect="equal",
    )
    axes.set_yticklabels(alphabet, rotation=0, ha="center", fontsize=7)
    sns.despine(ax=axes, right=False, top=False)

    axes = subplots[1]
    pos = "54"
    df = general_product[pos] * 100
    sns.heatmap(
        df,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_ax=cbar_axes,
        cbar_kws={"label": r"Decay factor (%)"},
    )
    axes.set(
        title="Position {}".format(pos),
        xlabel="Allele 1",
        ylabel="Allele 2",
        xticks=ticks,
        yticks=ticks,
        aspect="equal",
    )
    axes.set_yticklabels(alphabet, rotation=0, ha="center", fontsize=7)
    axes.set_xticklabels(alphabet, rotation=0, ha="center", fontsize=7)

    sns.despine(ax=axes, right=False, top=False)
    sns.despine(ax=cbar_axes, right=False, top=False)

    fig.tight_layout(h_pad=0.5)
    fig.savefig("figures/gb1.decay_factors.general_product.png", dpi=300)
    fig.savefig("figures/gb1.decay_factors.general_product.svg", dpi=300)

    # for pos, m in general_product.items():
    #     fig = sns.clustermap(m, cmap="Blues", vmin=0, vmax=1)
    #     fig.savefig("plots/gb1.general_product.pos{}.png".format(pos), dpi=300)
