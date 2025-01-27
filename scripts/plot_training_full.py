import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH, 2.3),
        # sharey=True,
    )
    ylims = {
        "smn1": (-115000, None),
        # "gb1": (-40000, None),
        "gb1": (None, -160000),
    }

    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]
    for dataset, axes in zip(["smn1", "aav", "gb1"], subplots):
        for kernel in ("Connectedness", "Jenga", "GeneralProduct"):
            max_value = -np.inf
            for i in range(1, 6):
                fpath = (
                    "output/{}.finetune.{}.{}.test_pred.csv.loss.csv".format(
                        dataset, i, kernel
                    )
                )
                if not exists(fpath):
                    continue

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

    subplots[0].set(ylabel="Marginal Log-Likelihood")
    # subplots[0].legend(loc=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/training_curves.png", format="png", dpi=300)
    print("Done")
