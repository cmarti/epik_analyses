import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    dataset = "aav"
    models = ["Exponential", "Connectedness", "Jenga", "GeneralProduct"]

    fig, subplots = plt.subplots(
        len(models),
        len(models),
        figsize=(FIG_WIDTH * 0.8, FIG_WIDTH * 0.8),
        sharey=True,
        sharex=True,
    )

    min_values = {k: np.inf for k in models}
    abs_min = np.inf
    for model1, row in zip(models, subplots):
        for model2, axes in zip(models, row):
            for i in range(1, 6):
                fpath = "output/{}.{}.{}.{}.test_pred.csv.loss.csv".format(
                    dataset, model2, i, model1
                )
                if not exists(fpath):
                    continue

                data = pd.read_csv(fpath, index_col=0)
                min_loss = data["loss"].min()
                axes.plot(
                    data["loss"],
                    c="grey",
                    lw=0.5,
                )

                shown = True
                if min_loss < min_values[model1]:
                    min_values[model1] = min_loss

                if min_loss < abs_min:
                    abs_min = min_loss

            # axes.set(ylim=(7000, 25000))
            if model1 == models[0]:
                axes.set_title("Initialization\n{}".format(model2))
            if model1 == models[-1]:
                axes.set(xlabel="Iteration number")
            if model2 == models[0]:
                axes.set(ylabel="{}\nLoss function".format(model1))

    for i, model in enumerate(models):
        for j in range(len(models)):
            subplots[i, j].axhline(
                min_values[model], lw=0.5, c="darkred", linestyle="--"
            )
            if np.isfinite(abs_min):
                subplots[i, j].axhline(
                    abs_min, lw=0.5, c="black", linestyle="--"
                )
    print(min_values)

    # fig.suptitle('Training model')
    # fig.supylabel('Initialization model')
    fig.tight_layout()
    fig.savefig(
        "plots/training_init.{}.png".format(dataset), format="png", dpi=300
    )
    print("Done")
