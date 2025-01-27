import pandas as pd
import matplotlib.pyplot as plt
from scripts.figures.plot_utils import plot_cv_curve, FIG_WIDTH
from scripts.figures.settings import MODELS


if __name__ == "__main__":
    dataset = "aav"
    metric = "r2"
    fraction_width = 0.65

    # Load data
    fpath = "results/{}.cv_curves.csv".format(dataset)
    data = pd.read_csv(fpath, index_col=0)
    data = data.loc[data['model'] != 'Variance Component', :]

    # Make figure
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * fraction_width, 0.35 * FIG_WIDTH))
    plot_cv_curve(axes, data, metric=metric)
    
    obs = data["model"].unique()
    hue_order = [x for x in MODELS if x in obs][::-1]
    handles, labels = axes.get_legend_handles_labels()
    ordered_handles = [handles[hue_order.index(label)] for label in hue_order[::-1]]
    axes.legend(ordered_handles, hue_order, loc=(1.05, 0.25))

    # Save figure
    fig.tight_layout()
    fig.savefig("figures/{}.{}.svg".format(dataset, metric), dpi=300)
    fig.savefig("figures/{}.{}.png".format(dataset, metric), dpi=300)
