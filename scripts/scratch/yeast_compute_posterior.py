#!/usr/bin/env python
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from epik.src.kernel import ConnectednessKernel
from epik.src.model import EpiK
from epik.src.utils import seq_to_one_hot


def get_yeast_seqs(loci=["ENA1", "HAL9", "MKT1"]):
    data = pd.read_csv("results/qtls_li_hq_results.csv", index_col=0)
    loci_idx = np.isin(data.index, loci)
    sorted_loci = data.index[loci_idx].values
    sorted_loci = ['bc'] + list(sorted_loci)

    seqs, labels = [], []
    for bc in "AB":
        for subseq in product(["A", "B"], repeat=len(loci)):
            seq = np.array([bc] * 83)
            seq[loci_idx] = np.array(subseq)
            seq = "".join(seq)
            seqs.append(seq)
            labels.append("".join((bc,) + subseq))
    return(sorted_loci, seqs, labels)


if __name__ == "__main__":
    fpath = 'datasets/qtls_li_hq.csv'
    data = pd.read_csv(fpath, index_col=0)
    X = seq_to_one_hot(data.index.values, alleles=['A', 'B'])
    y = torch.Tensor(data["y"].values)
    y_var = torch.Tensor(data["y_var"].values)
    
    fpath = "output_new/qtls_li_hq.Connectedness.2.model_params.pth"
    params = torch.load(fpath, map_location=torch.device('cpu'))
    
    kernel = ConnectednessKernel(n_alleles=2, seq_length=83, theta0=params['covar_module.theta'],
                                 log_var0=torch.Tensor(params['covar_module.log_var']))
    model = EpiK(kernel=kernel, train_noise=True)
    model.set_data(X, y, y_var)
    noise = params['likelihood.second_noise_covar.raw_noise']
    model.likelihood.second_noise_covar.raw_noise = torch.nn.Parameter(noise)
    
    
    loci, seqs, labels = get_yeast_seqs()
    X_pred = seq_to_one_hot(seqs, alleles=['A', 'B'])
    f = model.get_posterior(X_pred, calc_covariance=True)
    means = pd.DataFrame({'mean': f.mean.numpy()}, index=labels)
    cov = pd.DataFrame(f.covariance_matrix.numpy(), index=labels, columns=labels)
    
    means.to_csv('results/qtls_li_hq.Connectedness.2.post_means.csv')
    cov.to_csv('results/qtls_li_hq.Connectedness.2.post_cov.csv')
    
