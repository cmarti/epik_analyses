#!/usr/bin/env python

import gpytorch
import numpy as np
import pandas as pd
import torch
from epik.src.kernel import (
    ConnectednessKernel,
)
from epik.src.model import EpiK
from epik.src.utils import encode_seqs, get_random_sequences
from scipy.stats import multivariate_normal

if __name__ == "__main__":
    alphabet = ["A", "C", "G", "T"]
    n_alleles = len(alphabet)
    seq_length = 8

    records = []
    for n in [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 7500, 10000]:
        seqs = get_random_sequences(n=n, seq_length=seq_length, alphabet=alphabet)
        X = encode_seqs(seqs, alphabet=alphabet)
        y_var = 0.1 * np.ones(n)
        D = np.diag(y_var)
        with torch.no_grad():
            kernel = ConnectednessKernel(n_alleles=n_alleles, seq_length=seq_length)
            mu = np.zeros(n)
            Sigma = kernel(X, X).to_dense().numpy() + D

            gaussian = multivariate_normal(mu, Sigma)
            y = gaussian.rvs()
            logp1 = gaussian.logpdf(y)

            model = EpiK(kernel)
            model.set_data(X=X, y=y, y_var=y_var)

            with gpytorch.settings.num_trace_samples(100), gpytorch.settings.max_lanczos_quadrature_iterations(100):
                for _ in range(10):
                    logp2 = model.calc_mll().item()
                    records.append({"n": n, "true_logp": logp1, "logp": logp2})
                    print(records[-1])
    records = pd.DataFrame(records)
    records.to_csv('results/test_mll_simulations2.csv')
