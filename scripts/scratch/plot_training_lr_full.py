import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    ylims = {'smn1': (-30000, None),
             'aav': (-17000, None),
            #  'gb1': (-13000, None),
             }
    fig, subplots = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH, 2.3),
        # sharey=True,
    )

    cmap = cm.get_cmap('Blues')
    lrs = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    # lrs = ["0.001", "0.0005", "0.0001", "0.00005", "0.00001", "0.000005"]
    colors = [cmap(x) for x in np.linspace(0.1, 0.9, len(lrs))]
    for dataset, axes in zip(['smn1', 'aav', 'gb1'], subplots):
        max_value = -np.inf
        for lr, c in zip(lrs, colors):
            fpath = "output/{}.finetune.Adam.opt_lr.{}.loss.csv".format(dataset, lr)
            if not exists(fpath):
                continue
            print(fpath)

            data = pd.read_csv(fpath, index_col=0)
            if "mll" not in data:
                continue

            max_mll = data["mll"].max()
            axes.plot(
                data["mll"],
                c=c,
                label=str(lr),
                lw=1,
                alpha=1,
            )

            shown = True
            if max_mll > max_value:
                max_value = max_mll

        axes.axhline(
            max_value, lw=0.5, c="darkred",
            # label="Best model",
            linestyle="--"
        )
        axes.set(xlabel="Iteration number", title=dataset.upper(),
                 ylim=ylims.get(dataset, None),
                )
        print(dataset, max_value)
    
    subplots[0].set(ylabel="Marginal Log-Likelihood")
    subplots[0].legend(loc=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/learning_rates_full.png", format="png", dpi=300)
    print("Done")
