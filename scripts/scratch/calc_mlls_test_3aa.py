#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from epik.src.kernel import (
    AdditiveKernel,
    PairwiseKernel,
    VarianceComponentKernel,
    ExponentialKernel,
    ConnectednessKernel,
    JengaKernel,
    GeneralProductKernel,
)
import gpytorch
from scipy.stats import multivariate_normal
from epik.src.utils import seq_to_one_hot, encode_seqs, get_full_space_one_hot
from epik.src.model import EpiK
from gpmap.src.linop import KronOperator, MultivariateGaussian, KernelOperator


AMINOACIDS = [
    "R",
    "K",
    "E",
    "D",
    "Q",
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


if __name__ == "__main__":
    np.random.seed(0)
    dataset = "gb1"
    i = 1
    alleles = sorted(AMINOACIDS)
    n_alleles = len(alleles)
    seq_length = 4

    idx = [2, 3]
    l = len(idx)
    kernel = "Connectedness"
    with torch.no_grad():
        fpath = "output/{}.{}.{}.test_pred.csv.model_params.pth".format(
            dataset, i, kernel
        )
        params = torch.load(fpath, map_location=torch.device("cpu"))
        kernel = ConnectednessKernel(
            seq_length=l,
            n_alleles=n_alleles,
            log_var0=params["covar_module.module.log_var"],
            # theta0=params["covar_module.module.theta"][idx],
        )
        print("Kernel parameters")
        print(kernel.log_var, kernel.theta)

        sigma2_site = np.exp(
            kernel.log_var.detach().numpy() / kernel.seq_length
        )
        kernels = kernel.get_site_kernels().detach().numpy() * sigma2_site

        mu = np.zeros(n_alleles**l)
        K1 = KronOperator(kernels)
        gaussian = MultivariateGaussian(mu, K1)
        y = gaussian.sample(n_samples=1).flatten()
        logp1 = gaussian.logp(y)  # / y.shape[0]
        print("gpmap-tools mll = {}".format(logp1))

        # Compute with gpytorch
        # idx = np.random.randint(0, mu.shape[0], size=10)
        X = get_full_space_one_hot(l, n_alleles)
        K2 = kernel(X, X)

        # print("Testing submatrices")
        # k1 = KernelOperator(K1, idx, idx).todense()
        # k2 = kernel(X[idx], X[idx]).to_dense().numpy()
        # assert np.allclose(k1, k2, atol=1e-1)

        # # Compute with scipy
        # gaussian = multivariate_normal(mu, K2.detach().numpy())
        # logp3 = gaussian.logpdf(y) / y.shape[0]
        # print(logp3)

        print("With gpytorch MvGaussian")
        with gpytorch.settings.max_lanczos_quadrature_iterations(30):
            with gpytorch.settings.num_trace_samples(50):
                with gpytorch.settings.max_cholesky_size(100):
                    gaussian = gpytorch.distributions.MultivariateNormal(
                        torch.tensor(mu, dtype=torch.float32), K2
                    )

                    for _ in range(5):
                        logp4 = (
                            gaussian.log_prob(
                                torch.tensor(y, dtype=torch.float32)
                            ).item()
                            # / y.shape[0]
                        )
                        print(logp4)

                        # model = EpiK(kernel, device="cpu")
                        # model.set_data(X, y=y)
                        # logp2 = model.calc_mll().item()
                        # print(logp2, logp4)
