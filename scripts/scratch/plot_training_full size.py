import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH / 3.0, 2.3),
        # sharey=True,
    )
    ylims = {
        # "smn1": (-115000, None),
        # "gb1": (-40000, None),
        # "gb1": (None, -160000),
    }

    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]

    CMAPS = {
        "Connectedness": cm.get_cmap("Purples"),
        "Jenga": cm.get_cmap("Reds"),
        "GeneralProduct": cm.get_cmap('Greens'),
    }
    dataset = "gb1"
    total = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0).shape[0]
    for kernel in ("Connectedness", "Jenga", "GeneralProduct"):
        max_value = -np.inf
        # for i in range(1, 60):
        # for i in range(14, 31):
        js = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        js = [15, 25, 40]
        for i in js:
            fpath = "output/{}.{}.{}.test_pred.csv.loss.csv".format(
                dataset, i, kernel
            )
            if not exists(fpath):
                continue

            data = pd.read_csv(fpath, index_col=0)
            fpath = "splits/{}.{}.train.csv".format(dataset, i)
            train = pd.read_csv(fpath, index_col=0)

            if "mll" not in data:
                continue
            data["mll"] = data["mll"] / train.shape[0]

            max_mll = data["mll"].max()
            axes.plot(
                data["mll"],
                # c=MODELS_PALETTE[kernel],
                c=CMAPS[kernel](train.shape[0] / total),
                lw=0.5,
                alpha=1,
            )

            shown = True
            if max_mll > max_value:
                max_value = max_mll

        if np.isfinite(max_value):
            axes.axhline(
                max_value, lw=0.5, c=MODELS_PALETTE[kernel], linestyle="--"
            )
    axes.set(
        xlabel="Iteration number",
        ylabel="Marginal Log-Likelihood",
        title=dataset.upper(),
        ylim=ylims.get(dataset, None),
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/training_curves_size.png", format="png", dpi=300)
    print("Done")
