import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    dataset_labels = {
        "gb1": "Protein GB1",
        "aav": "AAV2 Capside",
        "smn1": "SMN1 5´splice site",
        "qtls_li_hq": "Yeast growth in Li",
    }
    ndatasets = len(dataset_labels)
    fig, subplots = plt.subplots(
        1,
        ndatasets,
        figsize=(FIG_WIDTH, 2.0),
        # sharey=True,
    )

    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]
    for (dataset, label), axes in zip(dataset_labels.items(), subplots):
        max_value = -np.inf
        for model, color in MODELS_PALETTE.items():
            if model not in (
                "Jenga",
                "Connectedness",
                "Exponential",
                "GeneralProduct",
            ):
                continue
            shown = False
            for i in range(1, 6):
                fpath = "output/{}.full.{}.{}.test_pred.csv.loss.csv".format(
                    dataset, i, model
                )
                if not exists(fpath):
                    continue

                data = pd.read_csv(fpath, index_col=0)
                if "mll" not in data:
                    continue

                max_mll = data["mll"].max()
                axes.plot(
                    data["mll"],
                    c=color,
                    label=model if not shown else None,
                    lw=0.5,
                )

                shown = True
                if max_mll > max_value:
                    max_value = max_mll

        axes.axhline(
            max_value, lw=0.5, c="darkred", label="Best model", linestyle="--"
        )
        axes.set(xlabel="Iteration number")
        axes.text(
            0.95,
            0.05,
            label,
            fontsize=8,
            ha="right",
            va="bottom",
            transform=axes.transAxes,
        )
    subplots[0].set(ylabel="Marginal Log-Likelihood")
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    subplots[0].legend(loc=(0, 1.05), ncol=6, fontsize=8)
    subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/training.png", format="png", dpi=300)
    print("Done")
