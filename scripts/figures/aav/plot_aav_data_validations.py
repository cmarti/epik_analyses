#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from scripts.settings import AAV, DATADIR
from scripts.figures.plot_utils import FIG_WIDTH, savefig


if __name__ == "__main__":
    dataset = AAV
    ref = 561
    charge = {"E": -1, "D": -1, "R": 1, "K": 1}

    print("Loading {} data".format(dataset))
    fpath = join(DATADIR, "{}.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)
    lims = data["y"].min(), data["y"].max()
    data["576"] = [x[576 - ref] for x in data.index]
    data["576_other"] = [
        x[: 576 - ref] + x[576 - ref + 1 :] for x in data.index
    ]
    data["569"] = [x[569 - ref] for x in data.index]
    data["569_other"] = [
        x[: 569 - ref] + x[569 - ref + 1 :] for x in data.index
    ]
    data["charge"] = [
        np.sum([charge.get(aa, 0) for aa in seq[-8:]])
        for seq in data.index.values
    ]
    xticks = np.array([-10, -5, 0, 5, 10])
    kwargs = {
        "alpha": 0.5,
        "bins": np.linspace(lims[0], lims[1], 50),
        "density": True,
    }

    fig, subplots = plt.subplots(
        3,
        1,
        figsize=(0.33 * FIG_WIDTH, 0.65 * FIG_WIDTH),
        height_ratios=(2, 3, 2),
    )

    print("Plotting phenotype distributions depending on site 569")
    axes = subplots[0]
    idx = data["569"] != "N"
    axes.hist(data.loc[idx, "y"], color="grey", label="Other", **kwargs)
    axes.hist(data.loc[~idx, "y"], color="black", label="569N", **kwargs)
    axes.legend(loc=0)
    axes.set(xlabel="DMS score", ylabel="Probability density", xticks=xticks)

    print("Plotting phenotype distributions depending on sites 579-588")
    axes = subplots[1]
    sns.violinplot(
        y="y",
        x="charge",
        data=data,
        color="grey",
        inner=None,
        linewidth=0.75,
        ax=axes,
    )
    axes.set(ylabel="DMS score", xlabel="579-588 charge")

    print("Plotting phenotype distributions depending on site 576")
    axes = subplots[2]
    idx = np.isin(data["576"], ["Y", "F", "W"])
    axes.hist(data.loc[~idx, "y"], color="grey", label="Other", **kwargs)
    axes.hist(data.loc[idx, "y"], color="black", label="576Y/F/W", **kwargs)
    axes.legend(loc=0)
    axes.set(xlabel="DMS score", ylabel="Probability density", xticks=xticks)

    fig.tight_layout(h_pad=1.5, pad=0.5)
    savefig(fig, "{}.validations".format(dataset))
