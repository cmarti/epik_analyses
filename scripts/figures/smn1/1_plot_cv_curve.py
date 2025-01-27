import pandas as pd
import matplotlib.pyplot as plt

from scripts.figures.plot_utils import plot_cv_curve, FIG_WIDTH


if __name__ == "__main__":
    dataset = "smn1"
    metric = "r2"

    # Load data
    fpath = "results/{}.cv_curves.csv".format(dataset)
    data = pd.read_csv(fpath, index_col=0)

    # Make figure

    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.4, FIG_WIDTH * 0.4))
    plot_cv_curve(axes, data, metric=metric)

    # Save figure
    fig.tight_layout()
    fig.savefig("figures/{}.{}.svg".format(dataset, metric), dpi=300)
    fig.savefig("figures/{}.{}.png".format(dataset, metric), dpi=300)
