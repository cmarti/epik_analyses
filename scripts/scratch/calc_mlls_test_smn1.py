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
from epik.src.utils import seq_to_one_hot, encode_seqs, get_full_space_one_hot
from epik.src.model import EpiK
from gpmap.src.linop import KronOperator, MultivariateGaussian, KernelOperator




if __name__ == "__main__":
    dataset = "smn1"
    i = 1
    alleles = ['A', 'C', 'G', 'U']
    n_alleles = len(alleles)
    seq_length = 8
    
    kernel = 'Connectedness'
    with torch.no_grad():
        fpath = "output/{}.{}.{}.test_pred.csv.model_params.pth".format(dataset, i, kernel)
        params = torch.load(fpath, map_location=torch.device("cpu"))
        kernel = ConnectednessKernel(seq_length=seq_length, n_alleles=n_alleles,
                                    log_var0=params['covar_module.log_var'],
                                    theta0=params['covar_module.theta'])
        log_var = params['covar_module.log_var'].detach().numpy()
        sigma2_site = np.exp(log_var / seq_length)
        kernels = kernel.get_site_kernels().detach().numpy() * sigma2_site
    
        
        mu = np.zeros(int(n_alleles ** seq_length))
        K1 = KronOperator(kernels)
        gaussian = MultivariateGaussian(mu, K1)
        y = gaussian.sample(n_samples=1).flatten()
        logp1 = gaussian.logp(y) / y.shape[0]
        print(logp1)
        
        # Compute with gpytorch
        X = get_full_space_one_hot(seq_length, n_alleles)
        kernel.use_keops = True
        with gpytorch.settings.num_trace_samples(100), gpytorch.settings.max_lanczos_quadrature_iterations(100), gpytorch.settings.max_preconditioner_size(0):
            model = EpiK(kernel, train_noise=False, preconditioner_size=0, device='cuda')
            model.set_data(X, y=y)
            logp2 = model.calc_mll()
            print(logp2)
