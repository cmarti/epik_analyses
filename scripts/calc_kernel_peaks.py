#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd

from epik.src.kernel.haploid import ARDKernel, RBFKernel
from epik.src.utils import seq_to_one_hot


def load_kernel(dataset, kernel):
    fpath = 'output/{}.{}.test_pred.csv.model_params.pth'.format(dataset, kernel)
    params = torch.load(fpath, map_location=torch.device('cpu'))
    logit_rho0 = params['covar_module.logit_rho']
    log_p0 = params['covar_module.log_p']
    kernel = ARDKernel(n_alleles=log_p0.shape[1],
                       seq_length=log_p0.shape[0],
                       logit_rho0=logit_rho0, log_p0=log_p0,
                       log_var0=0)
    # print('Loaded kernel with logit_rho: {}'.format(logit_rho0.cpu().numpy()))
    return(kernel)


if __name__ == '__main__':
    kernel_label = 'ARD'
    
    aa = sorted(['R', 'K', 'Q', 'E', 'D', 'N', 'H', 'S', 'T', 'A',
                 'V', 'I', 'L', 'M', 'P', 'G', 'Y', 'F', 'W', 'C'])
    nc = sorted(['A', 'C', 'G', 'U'])
    gb1_seqs2 = ['VDGV', 'WWLG', 'LICA']
    smn1_seqs2 = ['UCUUAAGU', 'CAGUAAGU', 'CAGUUCAA', 'ACUUAUCC',
                  'GGUCGUUU']
    
    for dataset, alleles, seqs2 in (('smn1', nc, smn1_seqs2),
                                    # ('gb1', aa, gb1_seqs2)
                                    ):
        seqs1 = pd.read_csv('output/{}.{}.test_pred.csv'.format(dataset, kernel_label), index_col=0).index.values
        x1 = seq_to_one_hot(seqs1, alleles)
        x2 = seq_to_one_hot(seqs2, alleles)

        # Compute kernel at all sequences
        kernel = load_kernel(dataset, kernel_label)
        
        K = np.vstack([kernel.forward(x1[i:i+10000], x2).detach().numpy()
                       for i in range(0, x1.shape[0], 10000)])
        K = pd.DataFrame(K, columns=seqs2, index=seqs1)
        print(K.loc[seqs2, :])
        K.to_csv('output/{}.{}_kernel_function.csv'.format(dataset, kernel_label))

        # kernel = load_RBF_kernel(dataset, id).to(device='cpu')   
        # K = kernel.forward(x1, x2).detach().cpu().numpy()
        # K = pd.DataFrame(K, columns=seqs2, index=seqs1)
        # print(K.loc[seqs2, :])
        # K.to_csv('output/{}.RBF_kernel_function.csv'.format(dataset, id))
