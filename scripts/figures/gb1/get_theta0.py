#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from os.path import exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE
from epik.src.kernel import SiteKernelAligner


if __name__ == "__main__":
    dataset = "gb1"
    aligner = SiteKernelAligner(n_alleles=20, seq_length=4)

    for i in range(1, 6):
        model = "Connectedness"
        fpath = "output/{}.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            dataset, i, model
        )
        params = torch.load(fpath, map_location="cpu")
        theta0 = params["covar_module.theta"].numpy()
        theta1 = torch.Tensor(aligner.connectedness_to_jenga(theta0))
        params["covar_module.theta"] = theta1
        suffix = "Jenga.model_params.pth"
        fpath = "output/{}.{}{}.{}".format(dataset, model, i, suffix)
        torch.save(params, fpath)

        model = "Jenga"
        suffix = "test_pred.csv.max_evid.model_params.pth"
        fpath = "output/{}.full.{}.{}.{}".format(dataset, i, model, suffix)
        params = torch.load(fpath, map_location="cpu")
        theta0 = params["covar_module.theta"].numpy()
        theta1 = torch.Tensor(aligner.jenga_to_general_product(theta0))
        params["covar_module.theta"] = theta1
        suffix = "GeneralProduct.model_params.pth"
        fpath = "output/{}.{}{}.{}".format(dataset, model, i, suffix)
        torch.save(params, fpath)
