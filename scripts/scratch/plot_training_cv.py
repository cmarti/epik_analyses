import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    
    datasets = ['smn1', 'gb1', 'aav', 'qtls_li_hq']
    
    kernels = [
        "Additive",
        "Pairwise",
        "Exponential",
        "VC",
        "Connectedness",
        "Jenga",
        "GeneralProduct",
    ]
    
    fig, subplots = plt.subplots(
        len(datasets),
        len(kernels),
        figsize=(2. * len(kernels), 1.15 * len(kernels)),
        sharey='row',  # sharex=True
    )

    cmap = cm.get_cmap("Blues")
    for dataset, ax_row in zip(datasets, subplots):    
        for kernel, axes in zip(kernels, ax_row):
            for i in range(1, 61):
                n = pd.read_csv(
                    "splits/{}.{}.train.csv".format(dataset, i), index_col=0
                ).shape[0]
                fpath = "output_new/{}.{}.{}.loss.csv".format(dataset, i, kernel)
                if not exists(fpath):
                    continue

                data = pd.read_csv(fpath, index_col=0)
                data["mll_avg"] = data["mll"] / n
                axes.plot(
                    data["mll_avg"],
                    lw=0.5,
                    alpha=0.2,
                    # c="black",
                    c=cmap(i / 60.0),
                )

            axes.text(
                0.95,
                0.05,
                kernel,
                ha="right",
                va="bottom",
                transform=axes.transAxes,
                fontsize=8,
            )
    subplots[0, 0].set_ylim((-10, -2))
    subplots[1, 0].set_ylim((-2, -0.5))
    subplots[2, 0].set_ylim((-4, -1.))
    subplots[3, 0].set_ylim((0.5, 1.25))

    # subplots[0, 0].set(ylabel="Marginal Log-Likelihood")
    # subplots[1, 0].set(ylabel="Marginal Log-Likelihood")
    # subplots[0].legend(loc=4)
    fig.supxlabel('Number of training iterations', x=0.525, ha='center')
    fig.supylabel('Point-wise Marginal log-likelihood',
                  x=0.015, y=0.5, ha='center', va='center')
    fig.tight_layout()
    # fig.subplots_adjust(top=0.8)
    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/training_curves_cv.png", format="png", dpi=300)
    print("Done")
