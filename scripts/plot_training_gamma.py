import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    ylims = {'smn1': (-30000, None),
             'aav': (-17000, None),
             'gb1': (-13000, None)}
    
    datasets = ['smn1', 'aav', 'gb1']
    n = len(datasets)
    fig, subplots = plt.subplots(
        1,
        n,
        figsize=(FIG_WIDTH, 2.3),
        # sharey=True,
    )
    if n == 1:
        subplots = [subplots]

    cmap = cm.get_cmap('Reds')
    gammas = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9]
    colors = [cmap(x) for x in np.linspace(0.2, 0.9, len(gammas))]
    for dataset, axes in zip(datasets, subplots):
        max_value = -np.inf
        for gamma, c in zip(gammas[::-1], colors[::-1]):
            gamma_mll = []
            for i in range(1, 4):
                fpath = "output/{}.Adam.opt_gamma.{}.{}.loss.csv".format(dataset, gamma, i)
                if not exists(fpath):
                    continue
                data = pd.read_csv(fpath, index_col=0)

                max_mll = data["mll"].max()
                gamma_mll.append(max_mll)
                axes.plot(
                    data["mll"],
                    c=c,
                    label=str(gamma) if i == 1 else None,
                    lw=1,
                    alpha=0.5,
                )

                shown = True
                if max_mll > max_value:
                    max_value = max_mll
            print(gamma, np.mean(gamma_mll), gamma_mll)
        
        for i in range(1, 4):
            fpath = "output/{}.Adam.opt_lr.{}.{}.loss.csv".format(dataset, 0.01, i)
            if not exists(fpath):
                print(fpath)
                continue
            data = pd.read_csv(fpath, index_col=0)

            max_mll = data["mll"].max()
            axes.plot(
                data["mll"],
                c='black',
                lw=1,
                alpha=0.5,
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
                #  ylim=(-30000, -22500),
                ylim=ylims[dataset],
                )
        print(dataset, max_value)
    
    subplots[0].set(ylabel="Marginal Log-Likelihood")
    subplots[0].legend(loc=4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8)

    # subplots[2].set_ylim((-30000, None))

    fig.savefig("plots/gammas.png", format="png", dpi=300)
    print("Done")
