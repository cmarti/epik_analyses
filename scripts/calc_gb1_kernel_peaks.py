#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from epik.src.kernel.haploid import ARDKernel, RBFKernel
from epik.src.utils import seq_to_one_hot,  get_full_space_one_hot


def load_ARD_kernel(dataset, id):
    fpath = 'output_gpu/{}.{}.ARD.test_pred.csv.model_params.pth'.format(dataset, id)
    params = torch.load(fpath)
    logit_rho0 = params['covar_module.logit_rho']
    log_p0 = params['covar_module.log_p']
    kernel = ARDKernel(n_alleles=20, seq_length=4,
                       logit_rho0=logit_rho0, log_p0=log_p0,
                       log_var0=0)
    print('Loaded ARD kernel with logit_rho: {}'.format(logit_rho0.cpu().numpy()))
    return(kernel)


def load_RBF_kernel(dataset, id):
    fpath = 'output_gpu/{}.{}.RBF.test_pred.csv.model_params.pth'.format(dataset, id)
    params = torch.load(fpath)
    logit_rho0 = params['covar_module.logit_rho']
    kernel = RBFKernel(n_alleles=20, seq_length=4, logit_rho0=logit_rho0, correlation=True)
    print('Loaded RBF kernel with logit_rho: {}'.format(logit_rho0.cpu().numpy()))
    return(kernel)


if __name__ == '__main__':
    id = '29'
    dataset = 'gb1'
    
    aa = sorted(['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                 'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'])
    seqs1 = pd.read_csv('output_gpu/gb1_landscape.csv', index_col=0).index.values
    seqs2 = ['VDGV', 'WWLG', 'LICA']
    x1 = seq_to_one_hot(seqs1, aa)
    x2 = seq_to_one_hot(seqs2, aa)

    kernel = load_ARD_kernel(dataset, id).to(device='cpu')   
    K = kernel.forward(x1, x2).detach().cpu().numpy()
    K = pd.DataFrame(K, columns=seqs2, index=seqs1)
    print(K.loc[seqs2, :])
    K.to_csv('output_gpu/{}.{}.ARD_kernel_function.csv'.format(dataset, id))

    kernel = load_RBF_kernel(dataset, id).to(device='cpu')   
    K = kernel.forward(x1, x2).detach().cpu().numpy()
    K = pd.DataFrame(K, columns=seqs2, index=seqs1)
    print(K.loc[seqs2, :])
    K.to_csv('output_gpu/{}.{}.RBF_kernel_function.csv'.format(dataset, id))
