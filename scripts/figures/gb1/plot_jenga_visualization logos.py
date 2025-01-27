#!/usr/bin/env python
import pandas as pd
import numpy as np

import logomaker as lm
import matplotlib.pyplot as plt

from scripts.figures.settings import POSITIONS

if __name__ == "__main__":
    plt.rcParams["font.family"] = "Arial"
    dataset = 'gb1'
    kernel_visualization = "Jenga"
    nrows, ncols = 1, 3

    # Load visualization
    nodes = pd.read_csv(
        "output/gb1.{}.nodes.csv".format(kernel_visualization), index_col=0
    )

    fig, subplots = plt.subplots(
        nrows,
        ncols,
        figsize=(7.5, 1.75),
        sharex=False,
        sharey=False,
    )

    idxs = [nodes['1'] > 0.3,
            nodes['2'] > 0.8,
            nodes['2'] < -0.8]
    for i, (idx, axes) in enumerate(zip(idxs, subplots)):
        seqs = nodes.index[idx].values
        m = lm.alignment_to_matrix(seqs, to_type='probability', pseudocount=0)
        lm.Logo(m, ax=axes, color_scheme='chemistry')

        axes.set(xlabel="Position", ylabel="", title='Region {}'.format(i + 1))
        axes.set_xticks(np.arange(4))
        axes.set_xticklabels(POSITIONS[dataset])

    subplots[0].set(ylabel="Frequency")
    fig.tight_layout(w_pad=0.05)
    fig.savefig("figures/gb1.visualization.logos.png", dpi=300)
    fig.savefig("figures/gb1.visualization.logos.svg", dpi=300)
