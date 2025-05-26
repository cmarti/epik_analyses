import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import exists
from scripts.figures.plot_utils import MODELS_PALETTE


if __name__ == "__main__":
    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]
    MODELS_PALETTE["VC"] = MODELS_PALETTE["Variance Component"]
    datasets = ['smn1', 'gb1', 'aav', 'qtls_li_hq']
    fig, subplots = plt.subplots(
        1,
        len(datasets),
        figsize=(2.5 * len(datasets), 2.3),
        # sharey=True,
    )

    for dataset, axes in zip(datasets, subplots):
        mlls = []
        total = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0).shape[0]
        for kernel in (
            'Additive',
            'Pairwise',
            "Exponential",
            "VC",
            "Connectedness",
            "Jenga",
            "GeneralProduct",
        ):
            max_value = -np.inf
            for i in range(1, 61):
                fpath = "output_new/{}.{}.{}.loss.csv".format(dataset, i, kernel)
                if not exists(fpath):
                    continue

                data = pd.read_csv(fpath, index_col=0)
                fpath = "splits/{}.{}.train.csv".format(dataset, i)
                train = pd.read_csv(fpath, index_col=0)

                mlls.append(
                    {
                        "mll": data["mll"].values[-10:].mean(),
                        "kernel": kernel,
                        "n": train.shape[0],
                    }
                )

        mlls = pd.DataFrame(mlls)
        mlls["avg_mll"] = mlls["mll"] / mlls["n"]
        mlls["logn"] = np.log10(mlls["n"])

        for kernel, df in mlls.groupby("kernel"):
            sns.regplot(
                x="logn",
                y="avg_mll",
                data=df,
                ax=axes,
                label=kernel,
                color=MODELS_PALETTE[kernel],
                lowess=True,
                scatter_kws={"s": 5, "lw": 0},
                line_kws={"lw": 0.75},
                ci=99,
            )
        axes.set(
            xlabel="# training seqs",
            xticks=[2, 3, 4, 5],
            ylabel='',
            xticklabels=[r"$10^2$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$"],
            title=dataset,
        )
    subplots[0].legend(loc=1)
    subplots[0].set(ylabel="Point-wise MLL",)
    fig.tight_layout()
    fig.savefig("plots/mll_vs_n.png", dpi=300)
