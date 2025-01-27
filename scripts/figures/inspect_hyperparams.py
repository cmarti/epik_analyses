#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch

from os.path import exists
from scipy.stats import pearsonr

from scripts.figures.settings import MODELS


if __name__ == "__main__":
    dataset, kernel, i = 'smn1', 'Jenga', 34
    
    fpath = "output/{}.opt_test_noise.{}.{}.test_pred.csv.model_params.pth".format(dataset, i, kernel)
    params = torch.load(fpath, map_location='cpu')
    print(params)
    
    fpath = "output/{}.opt_test_noise.{}.{}.test_pred.csv.max_evid.model_params.pth".format(dataset, i, kernel)
    params = torch.load(fpath, map_location='cpu')
    print(params)