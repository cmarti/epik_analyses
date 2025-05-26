#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from gpmap.linop import VjProjectionOperator, calc_vjs_variance_components
from gpmap.randwalk import WMWalk
import gpmap.plot.mpl as mplot
from scripts.figures.plot_utils import FIG_WIDTH
from itertools import combinations
from collections import defaultdict


def calc_variance_components(f, n_alleles, seq_length):
    vc = defaultdict(lambda: 0)
    marginal_sites = {}
    marginal_pw = {}
    total_variance = np.sum((f - f.mean()) ** 2)
    max_k = min(seq_length + 1, 6)
    for k in range(1, max_k):
        print(k)
        m_j = (n_alleles - 1) ** k
        vjs_k = defaultdict(lambda: 0)
        site_k = defaultdict(lambda: 0)
        vjs = calc_vjs_variance_components(f, a=n_alleles, l=seq_length, k=k)

        for j, lambda_j in vjs.items():
            vc[k] += lambda_j * m_j / total_variance

            for site in j:
                site_k[site] += lambda_j * m_j / total_variance

            if k > 1:
                for a, b in combinations(j, 2):
                    vjs_k[(a, b)] += lambda_j * m_j / total_variance

        if k > 1:
            marginal_pw[k] = vjs_k
        marginal_sites[k] = site_k

    marginal_pw = pd.DataFrame(marginal_pw).reset_index()
    cols = list(range(2, max_k))
    marginal_pw.columns = ["i", "j"] + cols
    marginal_pw["sum"] = marginal_pw[cols].sum(1)
    marginal_sites = pd.DataFrame(marginal_sites)
    vc = pd.DataFrame({"vc": pd.Series(vc)})

    return vc, marginal_sites, marginal_pw


if __name__ == "__main__":
    fpath = "output_new/qtls_li_hq.Connectedness.2.pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)

    with open("datasets/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]

    landscape.index = seqs
    pos_labels = np.append(
        ["BC"],
        np.load("datasets/qtls_li_hq.selected_loci.npy", allow_pickle=True),
    )
    seq_length = len(seqs[0])

    vc, marginal_sites, marginal_pw = calc_variance_components(
        landscape["coef"], 2, seq_length
    )
    print(marginal_pw)

    fig, subplots = plt.subplots(
        1,
        2,
        figsize=(1.1 * FIG_WIDTH, 0.43 * FIG_WIDTH),
        # figsize=(0.75 * FIG_WIDTH, 0.3 * FIG_WIDTH),
        sharex=True,
    )

    print("Plotting site marginal variance components in the MAP")
    axes = subplots[0]
    marginal_sites.index = pos_labels
    label = "% variance explained by\n interactions involving site $i$"
    sns.heatmap(
        np.log10(marginal_sites.T.iloc[::-1, :] * 100),
        ax=axes,
        cmap="Greys",
        cbar_kws={"label": label, "shrink": 0.8},
        vmin=-2,
        vmax=2,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(ylabel="Order of interaction $k$", xlabel="Locus")
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    print(
        "Plotting marginal variance components between pairs of sites in the MAP"
    )
    axes = subplots[1]
    label = "% variance explained by\ninteractions involving $i,j$"
    marginal_pw["high_order"] = marginal_pw["sum"] - marginal_pw[2]
    df1 = (
        pd.pivot_table(
            marginal_pw, index="i", columns="j", values="high_order"
        ).fillna(0)
        * 100
    )
    df2 = (
        pd.pivot_table(
            marginal_pw,
            index="i",
            columns="j",
            values=2,
        ).fillna(0)
        * 100
    )

    m = np.zeros((seq_length, seq_length))
    m[:-1, 1:] = df1.values
    m[1:, :-1] += df2.values.T
    df = pd.DataFrame(m, index=pos_labels, columns=pos_labels)

    sns.heatmap(
        np.log10(df),
        ax=axes,
        cmap="Greys",
        cbar_kws={
            "label": label,
            "shrink": 0.8,
        },
        vmin=-2,
        vmax=2,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(
        xlabel="Locus $i$",
        ylabel="Locus $j$",
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)

    fig.axes[-2].set_yticks([-2, -1, 0, 1, 2])
    fig.axes[-1].set_yticks([-2, -1, 0, 1, 2])
    fig.axes[-2].set_yticklabels([0.01, 0.1, 1, 10, 100])
    fig.axes[-1].set_yticklabels([0.01, 0.1, 1, 10, 100])
    sns.despine(left=False, bottom=False, right=False, top=False)
    axes.axline((0, 1), (0, -1), lw=0.75, linestyle="--")

    fig.tight_layout()
    fig.savefig("figures/yeast_landscape_variance_components.png", dpi=300)
