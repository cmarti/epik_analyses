import torch
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from os.path import exists
from epik.src.kernel import JengaKernel, GeneralProductKernel, ExponentialKernel, ConnectednessKernel


if __name__ == "__main__":
    max_value = -np.inf
    
    dataset = "smn1"
    models = ("Jenga", "GeneralProduct")
    kernels = (JengaKernel, GeneralProductKernel)
    
    # dataset = "qtls_li_hq"
    # models = ("Exponential", "Connectedness")
    # kernels = (ExponentialKernel, ConnectednessKernel)
    
    fig, subplots = plt.subplots(2, 2, figsize=(7, 7.0),
                                sharey=True,
                                gridspec_kw={'width_ratios': (0.025, 1)})
    
    for model, kernel, row in zip(models, kernels, subplots):
        models_params = []
        models_mlls = []
        for i in range(1, 31):
            fpath = (
                "output/{}.opt_test_noise.{}.{}.test_pred.csv.model_params.pth".format(
                    dataset, i, model
                )
            )
            if not exists(fpath):
                continue
            
            params = torch.load(fpath, map_location="cpu")
            # print(params['likelihood.second_noise_covar.raw_noise'], params['covar_module.log_var'], params['covar_module.theta'])
            # continue
            theta0 = params["covar_module.theta"]
            k = kernel(n_alleles=4, seq_length=8, theta0=theta0)
            models_params.append(k.get_site_kernels().detach().numpy().flatten())
            
            fpath = (
                "output/{}.opt_test_noise.{}.{}.test_pred.csv.loss.csv".format(
                    dataset, i, model
                )
            )
            df = pd.read_csv(fpath, index_col=0)
            models_mlls.append(df['mll'].values[-1])
            
        idx = np.argsort(models_mlls)
        models_mlls = np.array(models_mlls)[idx]
        models_params = np.array(models_params)[idx]

        
        axes = row[1]
        sns.heatmap(models_params, cmap='Blues', ax=axes,
                    cbar_kws={'label': 'Correlation'})
        axes.set(xlabel='Parameter', ylabel=None, xticklabels=[], title=model)
        fig.tight_layout()
        
        axes = row[0]
        sns.heatmap(np.expand_dims(models_mlls, 1),
                    cmap='Greys', ax=axes, cbar=False)
        axes.set(xlabel=None, ylabel='Training replicate')
        axes.set_xticklabels(['MLL'])
    
    fig.savefig("plots/training_test.params.png", format="png", dpi=300)
    print("Done")
