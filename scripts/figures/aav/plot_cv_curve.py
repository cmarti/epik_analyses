#!/usr/bin/env python
import matplotlib.pyplot as plt

from scripts.utils import load_r2
from scripts.settings import AAV, MODELS
from scripts.figures.plot_utils import plot_cv_curve, FIG_WIDTH, savefig


if __name__ == "__main__":
    dataset = AAV
    metric = "r2"
    p = 0.65

    data = load_r2(dataset)
    data = data.loc[data['model'] != 'Variance Component', :]

    print("Plotting R2 curves across models")
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.65, 0.35 * FIG_WIDTH))
    plot_cv_curve(axes, data, metric=metric)
    obs = data["model"].unique()
    hue_order = [x for x in MODELS if x in obs][::-1]
    handles, labels = axes.get_legend_handles_labels()
    ordered_handles = [handles[hue_order.index(label)] for label in hue_order[::-1]]
    axes.legend(ordered_handles, hue_order, loc=(1.05, 0.25))
    
    fig.tight_layout()
    savefig(fig, "{}.{}".format(dataset, metric))
