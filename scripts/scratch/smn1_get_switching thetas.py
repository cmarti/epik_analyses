#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE
from epik.src.kernel import SiteKernelAligner


if __name__ == "__main__":
    dataset = "smn1"
    aligner = SiteKernelAligner(n_alleles=4, seq_length=8)

    for i in range(1, 6):
        model = "Exponential"
        fpath = "output/{}.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            dataset, i, model
        )
        params = torch.load(fpath, map_location="cpu")
        fpath = "output/{}.{}{}.{}.model_params.pth".format(
            dataset, model, i, model
        )
        torch.save(params, fpath)
        theta1 = params["covar_module.module.theta"].numpy()

        fpath = "output/{}.{}{}.Connectedness.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.exponential_to_connectedness(theta1)
        ).flatten()
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.Jenga.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.exponential_to_jenga(theta1)
        )
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.GeneralProduct.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.exponential_to_general_product(theta1)
        )
        torch.save(params, fpath)

        model = "Connectedness"
        fpath = "output/{}.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            dataset, i, model
        )
        params = torch.load(fpath, map_location="cpu")
        fpath = "output/{}.{}{}.{}.model_params.pth".format(
            dataset, model, i, model
        )
        torch.save(params, fpath)

        theta1 = params["covar_module.module.theta"].unsqueeze(1).numpy()

        fpath = "output/{}.{}{}.Exponential.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.connectedness_to_exponential(theta1)
        )[0]
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.Jenga.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.connectedness_to_jenga(theta1)
        )
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.GeneralProduct.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.connectedness_to_general_product(theta1)
        )
        torch.save(params, fpath)

        model = "Jenga"
        fpath = "output/{}.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            dataset, i, model
        )
        params = torch.load(fpath, map_location="cpu")
        fpath = "output/{}.{}{}.{}.model_params.pth".format(
            dataset, model, i, model
        )
        torch.save(params, fpath)

        theta1 = params["covar_module.module.theta"].numpy()

        fpath = "output/{}.{}{}.Exponential.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.jenga_to_exponential(theta1)
        )[0]
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.Connectedness.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.jenga_to_connectedness(theta1)
        ).flatten()
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.GeneralProduct.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.jenga_to_general_product(theta1)
        )
        torch.save(params, fpath)

        model = "GeneralProduct"
        fpath = "output/{}.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            dataset, i, model
        )
        params = torch.load(fpath, map_location="cpu")
        fpath = "output/{}.{}{}.{}.model_params.pth".format(
            dataset, model, i, model
        )
        torch.save(params, fpath)

        theta1 = params["covar_module.module.theta"].numpy()

        fpath = "output/{}.{}{}.Exponential.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.general_product_to_exponential(theta1)
        )[0]
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.Connectedness.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.general_product_to_connectedness(theta1)
        ).flatten()
        torch.save(params, fpath)

        fpath = "output/{}.{}{}.Jenga.model_params.pth".format(
            dataset, model, i
        )
        params["covar_module.module.theta"] = torch.Tensor(
            aligner.general_product_to_jenga(theta1)
        )
        torch.save(params, fpath)
