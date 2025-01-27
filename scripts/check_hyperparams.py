import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join, exists
from scripts.figures.plot_utils import FIG_WIDTH, MODELS_PALETTE


if __name__ == "__main__":
    model = "Connectedness"
    for i in range(1, 6):
        print("Training {}".format(i))
        fpath1 = "output/smn1.full.{}.{}.test_pred.csv.max_evid.model_params.pth".format(
            i, model
        )
        params1 = torch.load(fpath1, map_location="cpu")
        fpath2 = "output/smn1.full.{}.{}.test_pred.csv.model_params.pth".format(
            i, model
        )
        params2 = torch.load(fpath2, map_location="cpu")

        print(params1)
        print(params2)
