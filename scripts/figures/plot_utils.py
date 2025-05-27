#!/usr/bin/env python
import numpy as np
import matplotlib.patches as patches
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import join
from scipy.stats import pearsonr
from scripts.settings import REF_SEQS, MODELS, ORDER, FIGDIR, POSITIONS

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
        
def highlight_mut_heatmap(axes, matrix, dataset, position):
    seq = REF_SEQS[dataset]
    positions = POSITIONS[dataset]
    pos_alleles = dict(zip(positions, seq))

    axes.set_clip_on(False)
    y = matrix.columns.tolist().index(pos_alleles[position])
    x = y
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
        subset = data[data["model"] == model]
        sns.lineplot(
            x="p_training",
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
            errorbar="sd",
        )

    hue_order = [x for x in MODELS if x in obs][::-1]
    handles, labels = axes.get_legend_handles_labels()
    idxs = [labels.index(label) for label in hue_order]
    ordered_handles = [handles[i] for i in idxs]
    axes.legend(ordered_handles, hue_order, loc=4, frameon=False, ncol=1)

    axes.set(
        # aspect=1 / 0.6,
        aspect=1,
        xlabel="Proportion of training data",
        ylabel=ylabels[metric],
        ylim=(0.0, 1) if metric == "r2" else (None, None),
        xlim=(0, 1),
    )


def plot_decay_rates(axes, decay_rates, dataset, **kwargs):
    sns.heatmap(
        decay_rates.T * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": r"Decay factor (%)"},
        **kwargs
    )
    axes.set(
        title="Jenga",
        xlabel="Position",
        ylabel="Allele",
        xticks=np.arange(decay_rates.shape[0]) + 0.5,
        yticks=np.arange(decay_rates.shape[1]) + 0.5,
    )
    
    rows, columns = decay_rates.index.values, decay_rates.columns.values
    axes.set_xticklabels(rows, rotation=0, ha="center", fontsize=7)
    
    if decay_rates.shape[1] > 1:
        axes.set_yticklabels(columns, rotation=0, ha="center", fontsize=7)
        highlight_seq_heatmap(axes, decay_rates, dataset=dataset)
    sns.despine(ax=axes, right=False, top=False)


def plot_mutation_decay_rates(axes, decay_rates, position, dataset, **kwargs):
    sns.heatmap(
        decay_rates * 100,
        ax=axes,
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": r"Decay factor (%)"},
        **kwargs
    )
    axes.set(
        title="Position {}".format(position),
        xlabel="Allele 1",
        ylabel="Allele 2",
        xticks=np.arange(decay_rates.shape[1]) + 0.5,
        yticks=np.arange(decay_rates.shape[0]) + 0.5,
        aspect="equal",
    )
    
    rows, columns = decay_rates.index.values, decay_rates.columns.values
    axes.set_xticklabels(rows, rotation=0, ha="center")
    axes.set_yticklabels(columns, rotation=0, ha="center")
    highlight_mut_heatmap(axes, decay_rates, dataset, position)
    sns.despine(ax=axes, right=False, top=False)


def plot_2D_hist(x, y, axes, vmin=0, vmax=3):
    r2 = pearsonr(x, y)[0] ** 2
    rmse = np.sqrt(np.mean((x - y) ** 2))

    lims = min(x.min(), y.min()), max(x.max(), y.max())
    bins = np.linspace(lims[0], lims[1], 100)
    diff = lims[1] - lims[0]
    lims = (lims[0] - 0.05 * diff, lims[1] + 0.05 * diff)

    H, xbins, ybins = np.histogram2d(x=x, y=y, bins=bins)
    with np.errstate(divide='ignore'):
        im = axes.imshow(
            np.log10(H.T[::-1, :]),
            cmap="viridis",
            extent=(xbins[0], xbins[-1], ybins[0], ybins[-1]),
            vmin=vmin,
            vmax=vmax,
        )
    axes.plot(lims, lims, lw=0.5, linestyle="--", c="black")
    axes.text(
        0.95,
        0.05,
        "$R^2$={:.2f}\nRMSE={:.2f}".format(r2, rmse),
        transform=axes.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ticks = [0, 50, 100, 150]
    axes.set(
        xlabel=r"Predicted PSI (%)",
        ylabel=r"Observed PSI (%)",
        xlim=lims,
        ylim=lims,
        aspect="equal",
        xticks=ticks,
        yticks=ticks,
    )
    return im

    
def savefig(fig, fname, save_svg=True, dpi=300):
    fpath = join(FIGDIR, fname)
    print('Saving figure to {}'.format(fpath))
    fig.savefig("{}.png".format(fpath), dpi=dpi)
    if save_svg:
        fig.savefig("{}.svg".format(fpath), dpi=dpi)
