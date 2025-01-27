#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions.transforms import CorrCholeskyTransform

import matplotlib.pyplot as plt
import gpmap.src.plot.mpl as plot
import logomaker as lm


from os.path import join
from gpmap.src.genotypes import select_genotypes
from gpmap.src.plot.mpl import get_hist_inset_axes
from gpmap.src.utils import read_edges
from gpmap.src.datasets import DataSet


if __name__ == "__main__":
    fname = "gb1.52.GeneralProduct.test_pred.csv.model_params.pth"
    fpath = join("output", fname)
    alleles_order = [
        "R",
        "K",
        "Q",
        "E",
        "D",
        "N",
        "H",
        "S",
        "T",
        "A",
        "V",
        "I",
        "L",
        "M",
        "P",
        "G",
        "Y",
        "F",
        "W",
        "C",
    ]
    alleles = sorted(alleles_order)

    params = torch.load(fpath, map_location="cpu")
    theta = params["covar_module.module.base_kernel.theta"]
    idx = params["covar_module.module.base_kernel.idx"]

    n_alleles = 20
    transform = CorrCholeskyTransform()

    fig, subplots = plt.subplots(2, 2, figsize=(9, 8))
    subplots = subplots.flatten()

    for i, axes in enumerate(subplots):
        # L = transform(theta[i])
        # K = L @ L.T
        v = -torch.exp(theta[i])
        log_cov = torch.zeros(
            (n_alleles, n_alleles), dtype=theta.dtype, device=theta.device
        )
        log_cov[idx[0], idx[1]] = v
        log_cov[idx[1], idx[0]] = v
        K = torch.exp(log_cov)

        K = pd.DataFrame(K, index=alleles, columns=alleles)[alleles_order].loc[
            alleles_order, :
        ]
        sns.heatmap(K, ax=axes, cmap="Blues", vmin=0, vmax=1)
        ticks = np.arange(20) + 0.5
        axes.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=alleles_order,
            title="Position {}".format(i + 1),
        )
        axes.set_yticklabels(alleles_order, rotation=0)

    fig.savefig("plots/gb1.general_product.png", dpi=300)
