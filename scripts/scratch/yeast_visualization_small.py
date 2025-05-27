#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gpmap.genotypes import select_genotypes
from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
import gpmap.plot.mpl as mplot


if __name__ == "__main__":
    np.random.seed(0)
    means = pd.read_csv('results/qtls_li_hq.Connectedness.2.post_means.csv', index_col=0)
    # seqs = [label for label in means.index if label.startswith("A")]
    # means = means.loc[seqs, :]
    # means.index = [x[1:] for x in means.index]
    print(means)
    cov = pd.read_csv('results/qtls_li_hq.Connectedness.2.post_cov.csv', index_col=0)
    # cov = cov.loc[seqs, :][seqs]
    b = np.array([1, -1, -1, 1, -1, 1, 1, -1])
    b = np.append(b * 0.5, 0.5 * b)
    epi_coeff3 = np.dot(b, means)
    epi_coeff3_sd = np.sqrt(np.dot(b, cov @ b))
    print(epi_coeff3_sd)
    print("Epi coeff3: {} [{}, {}]".format(epi_coeff3, epi_coeff3 -2 * epi_coeff3_sd, epi_coeff3 + 2 * epi_coeff3_sd))
    
    exit()
    
    # fpath = "output_new/qtls_li_hq.Connectedness.2.pred.csv"
    # landscape = pd.read_csv(fpath, index_col=0)

    with open("datasets/qtls_li_hq.seqs_key.txt") as fhand:
        seqs = [line.strip() for line in fhand]

    ref = "AAA"
    ws = [0.4, 0.6, 0.3]
    d = [
        np.sum([w * (c1 != c2) for c1, c2, w in zip(seq, ref, ws)])
        for seq in means.index
    ]


    fig, subplots = plt.subplots(2, 5, figsize=(2 * 4, 2 * 2.), sharex=True, sharey=True)
    subplots = subplots.flatten()
    L = np.linalg.cholesky(cov.values)
    for i in range(10):
        axes = subplots[i]
        sample = means['mean'].values + L @ np.random.normal(size=means.shape[0])
        space = SequenceSpace(X=means.index.values, y=sample)
        nodes = pd.DataFrame({'d': d, 'coef': sample}, index=means.index)
        edges = space.get_edges_df()
        mplot.plot_edges(axes, nodes, x="d", y="coef", edges_df=edges, alpha=0.5)
        mplot.plot_nodes(
            axes,
            nodes,
            x="d",
            y="coef",
            size=7.5,
            color="black",
            alpha=0.75
        )

        # kwargs = {
        #     "fmt": "o",
        #     "lw": 0,
        #     "elinewidth": 1,
        #     "capsize": 2,
        #     "markersize": 2.5,
        #     "color": "black",
        # }
        # axes.errorbar(
        #     x=landscape["d"],
        #     y=landscape["coef"],
        #     yerr=2 * landscape["stderr"],
        #     **kwargs,
        # )
        axes.set(
            xticks=[],
            xlim=(-0.25, 1.5),
            # title="ENA1-HAL9-MKT1 in RM background",
        )
    subplots[0].set_ylabel("Fitness")
    subplots[5].set_ylabel("Fitness")
    subplots[2].set_title("ENA1-HAL9-MKT1 in RM background")
    fig.supxlabel('Genotype', fontsize=10, x=0.55, y=0.05)
    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/yeast_visualization_small.png", dpi=300)
