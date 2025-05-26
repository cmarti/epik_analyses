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
from epik.src.utils import seq_to_one_hot, encode_seqs
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
    dataset = "gb1"
    i = 1
    alleles = sorted(AMINOACIDS)
    n_alleles = len(alleles)
    seq_length = 4
    
    fpath = 'output/gb1.Jenga.test_pred.csv'
    landscape = pd.read_csv(fpath, index_col=0)
    y = landscape['coef'].values
    
    kernel = 'Connectedness'
    with torch.no_grad():
        fpath = "output/{}.{}.{}.test_pred.csv.model_params.pth".format(dataset, i, kernel)
        params = torch.load(fpath, map_location=torch.device("cpu"))
        kernel = ConnectednessKernel(seq_length=seq_length, n_alleles=n_alleles,
                                    log_var0=params['covar_module.module.log_var'],
                                    theta0=params['covar_module.module.theta'])
        log_var = params['covar_module.module.log_var'].detach().numpy()[0]
        sigma2_site = np.exp(log_var / seq_length)
        kernels = kernel.get_site_kernels().detach().numpy() * sigma2_site
        
        mu = np.zeros_like(y)
        K1 = KronOperator(kernels)
        gaussian = MultivariateGaussian(mu, K1)
        logp1 = gaussian.logp(y) / y.shape[0]
        print(logp1)
        
        # Ensure we are encoding the same kernel matrix: look at a random submatrix
        x = np.random.randint(low=0, high=mu.shape[0] - 1, size=100)
        m1 = KernelOperator(K1, x1=x, x2=x).todense()
        seqs = landscape.index.values[x]
        X = encode_seqs(seqs, alphabet=alleles)
        m2 = kernel(X, X).to_dense()
        assert(np.allclose(m1, m2))
        
        # Compute with gpytorch
        fpath = "datasets/{}.seqs.txt".format(dataset)
        seqs = [line.strip() for line in open(fpath)]
        X = seq_to_one_hot(seqs, alleles)
        # kernel.use_keops = True
        with gpytorch.settings.num_trace_samples(100), gpytorch.settings.max_lanczos_quadrature_iterations(100), gpytorch.settings.max_preconditioner_size(0):
            model = EpiK(kernel, train_noise=False, preconditioner_size=0, device='cuda')
            model.set_data(X, y=y)
            logp2 = model.calc_mll()
            print(logp2)
