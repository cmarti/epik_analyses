#!/usr/bin/env python
import matplotlib.patches as patches
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scripts.figures.settings import REF_SEQS, MODELS, ORDER

FIG_WIDTH = 7

# Fonts
plt.rcParams["font.family"] = "Nimbus Sans"
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["legend.labelspacing"] = 0.15

plt.rcParams["axes.titlepad"] = 3

# Linewidths
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.width"] = 0.35
plt.rcParams["ytick.minor.width"] = 0.35

# COLORS = [
#     "silver",
#     "salmon",
#     "gray",
#     "cadetblue",
#     "violet",
#     "mediumslateblue",
#     "darkslateblue",
#     "black",
# ]
# MODELS_PALETTE = dict(zip(MODELS, COLORS))

grays = cm.get_cmap("binary")
blues = cm.get_cmap("Blues")
viridis = cm.get_cmap("viridis")

MODELS_PALETTE = {
    "Additive": grays(0.2),
    "Global epistasis": grays(0.4),
    "Pairwise": grays(0.6),
    "Variance Component": grays(0.8),
    "Exponential": blues(0.2),
    "Connectedness": blues(0.4),
    "Jenga": blues(0.6),
    "General Product": blues(0.8),
}


def highlight_seq_heatmap(axes, matrix, dataset):
    seq = REF_SEQS[dataset]

    axes.set_clip_on(False)
    for x, c in enumerate(seq):
        y = matrix.columns.tolist().index(c)
        axes.add_patch(
            patches.Rectangle(
                xy=(x, y),
                width=1.0,
                height=1.0,
                lw=0.75,
                fill=False,
                edgecolor="black",
                zorder=2,
            )
        )


def plot_cv_curve(axes, data, metric="r2", lw=0.8):
    ylabels = {
        "r2": "Test $R^2$",
        "rmse": "Test RMSE",
        "logit_r2": r"$\log_2\left(\frac{V_{model}}{V_{res}}\right)$",
        "log_likelihood": "Test log(L)",
    }

    obs = data["model"].unique()
    order = [x for x in ORDER if x in obs]
    for model in order:
        subset = data[data['model'] == model]
        sns.lineplot(x='p_training',
                     y=metric,
                     data=subset,
                     label=model,
                     color=MODELS_PALETTE[model],
                     lw=lw,
                     hue_order=order[::-1],
                     ax=axes,
                     err_style="bars",
                     err_kws={
                        "capsize": lw * 0.2,
                        "capthick": 0,
                        "lw": lw,
                        "elinewidth": 0.5,
                      },
                      errorbar="sd")
    
    hue_order = [x for x in MODELS if x in obs][::-1]
    handles, labels = axes.get_legend_handles_labels()
    ordered_handles = [handles[hue_order.index(label)] for label in hue_order[::-1]]
    axes.legend(ordered_handles, hue_order, loc=4, frameon=False, ncol=1)
    
    axes.set(
        # aspect=1 / 0.6,
        aspect=1,
        xlabel="Proportion of training data",
        ylabel=ylabels[metric],
        ylim=(0.0, 1) if metric == "r2" else (None, None),
        xlim=(0, 1),
    )
