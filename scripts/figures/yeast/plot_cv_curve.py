import pandas as pd
import matplotlib.pyplot as plt
from scripts.figures.plot_utils import plot_cv_curve


if __name__ == "__main__":
    dataset = "qtls_li_hq"
    metric = "r2"

    # Load data
    fpath = "results/{}.cv_curves.csv".format(dataset)
    data = pd.read_csv(fpath, index_col=0)

    # Make figure
    fig, axes = plt.subplots(1, 1, figsize=(3.0, 2.5))
    plot_cv_curve(axes, data, metric=metric)
    axes.legend(loc=4, fontsize=8)

    # Save figure
    fig.tight_layout()
    fig.savefig("figures/{}.{}.svg".format(dataset, metric), dpi=300)
    fig.savefig("figures/{}.{}.png".format(dataset, metric), dpi=300)
