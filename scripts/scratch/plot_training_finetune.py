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
        figsize=(FIG_WIDTH, 2.3),
        # sharey=True,
    )
    ylims = {
        "smn1": (-115000, None),
        # "gb1": (-40000, None),
        "gb1": (None, -160000),
    }

    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]
    dataset = "gb1"
    n = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0).shape[0]
    lrs = [0.001, "0.0001", "0.00005", "0.00002", "0.00001"]
    for lr in lrs:
        for kernel in ("Connectedness", "Jenga", "GeneralProduct"):
            max_value = -np.inf
            for i in range(1, 4):
                fpath = (
                    "output/{}.finetune.{}.{}.{}.test_pred.csv.loss.csv".format(
                        dataset, lr, i, kernel
                    )
                )
                if not exists(fpath):
                    continue
                print(fpath)

                data = pd.read_csv(fpath, index_col=0)
                if "mll" not in data:
                    continue

                max_mll = data["mll"].max()
                axes.plot(
                    data["mll"],
                    c=MODELS_PALETTE[kernel],
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
            print(dataset, kernel, max_value)
        axes.set(
            xlabel="Iteration number",
            title=dataset.upper(),
            ylim=ylims.get(dataset, None),
        )

    axes.set(ylabel="Marginal Log-Likelihood")
    # subplots[0].legend(loc=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/training_curves_lr.png", format="png", dpi=300)
    print("Done")
