#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk
import gpmap.plot.mpl as mplot
import logomaker


def annotate_seq(
    axes,
    seq,
    label,
    df,
    dx,
    dy,
    ha,
    va,
    x="1",
    y="2",
    fontsize=8,
    arrow_size=0.75,
):
    x, y = df.loc[seq, [x, y]]
    axes.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=0.3 * arrow_size,
            headwidth=4 * arrow_size,
            headlength=6 * arrow_size,
        ),
        ha=ha,
        va=va,
        fontsize=fontsize,
    )


if __name__ == "__main__":
    data = pd.read_csv("datasets/aav.csv", index_col=0)
    m = logomaker.alignment_to_matrix(
        data.index.values, to_type="probability", pseudocount=0
    )

    fig, axes = plt.subplots(1, 1, figsize=(4.5, 1.5))
    logomaker.Logo(m, ax=axes, color_scheme="chemistry")
    fig.tight_layout(w_pad=0.2)
    fig.savefig("figures/aav_logo.png", dpi=300)
