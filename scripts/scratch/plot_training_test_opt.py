import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    fig, subplots = plt.subplots(
        1,
        4,
        figsize=(FIG_WIDTH, 2.0),
        sharey=True,
    )

    MODELS_PALETTE["GeneralProduct"] = MODELS_PALETTE["General Product"]
    dataset = 'smn1'
    max_value = -np.inf
    models = ('Exponential', 'Connectedness', 'Jenga', 'GeneralProduct')
    # models = ('Jenga', 'GeneralProduct')
    for model, axes in zip(models, subplots):
        for i in range(1, 61):
            fpath = "output/{}.opt_test_noise.{}.{}.test_pred.csv.loss.csv".format(
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
                c='grey' if i < 46 else 'darkred',
                lw=0.5,
                alpha=0.5,
            )

            shown = True
            if max_mll > max_value:
                max_value = max_mll

        axes.axhline(
            max_value, lw=0.5, c="darkred", label="Best model", linestyle="--"
        )
        axes.set(xlabel="Iteration number",
                #  ylim=(-30000, -22500),
                ylim=(-30000, -15000),
                 )
        axes.text(
            0.95,
            0.05,
            model,
            fontsize=8,
            ha="right",
            va="bottom",
            transform=axes.transAxes,
        )
        print(model, max_value)
    
    subplots[0].set(ylabel="Marginal Log-Likelihood")
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)


    subplots[0].legend(loc=(0, 1.05), ncol=6, fontsize=8)
    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/{}.training_test.png".format(dataset), format="png", dpi=300)
    print("Done")
