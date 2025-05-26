#!/usr/bin/env python
import gpytorch
import pandas as pd
import numpy as np
import torch
from epik.src.kernel import ExponentialKernel, ConnectednessKernel
from epik.src.model import EpiK
from epik.src.utils import seq_to_one_hot


if __name__ == "__main__":
    np.random.seed(0)
    print("Loading data")
    i = 1
    alphabet = ["A", "B"]
    alleles = sorted(alphabet)
    n_alleles = len(alleles)
    seq_length = 83
    dataset = 'qtls_li_hq'
    data = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0)
    data = data.loc[np.random.uniform(size=data.shape[0]) < 0.2, :]

    seqs = data.index.values
    X = seq_to_one_hot(seqs, alleles)
    y = data["y"].values
    print(y.var())
    # exit()
    y_var = data["y_var"].values

    # Load trained parameters
    kernel = ExponentialKernel(seq_length=seq_length, n_alleles=n_alleles, log_var0=torch.tensor(0.0))
    model = EpiK(kernel, train_noise=True)
    model.set_data(X, y=y, y_var=y_var)
    fpath = "output_new/{}.full.1.Exponential.model_params.pth".format(dataset)
    params = torch.load(fpath, map_location='cpu')
    print(params)
    # model.set_params(params)
    
    records = []
    for theta in np.linspace(-13, -8, 50):
        for log_var in np.linspace(-2, 3, 50):
            with torch.no_grad():
                with gpytorch.settings.max_lanczos_quadrature_iterations(100):
                    with gpytorch.settings.max_cholesky_size(10000):
                        params['covar_module.theta'] = torch.Tensor([[theta]])
                        params['covar_module.log_var'] = torch.Tensor([log_var])
                        model.set_params(params)
                        mll = model.calc_mll().item()
                        records.append({"theta": theta, 'log_var': log_var, 'mll': mll})
                        # print(model.kernel(X[:3], X[:3]).to_dense(), mll)
                        print(records[-1])
    records = pd.DataFrame(records)
    records.to_csv('output/exponential_yeast_grid_search.csv')
