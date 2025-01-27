from os.path import join

import pandas as pd
import numpy as np
import torch

from scripts.figures.settings import ALPHABET, POSITIONS
from torch.distributions.transforms import CorrCholeskyTransform


def theta_to_decay_rates(theta, kernel, positions, alphabet):
    alleles = sorted(alphabet)
    n_alleles = len(alphabet)

    if kernel == "Exponential":
        log_rho = theta
        rho = torch.exp(log_rho)
        # print(rho, (1 - rho) / (1 + (n_alleles - 1) * rho))
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        delta = pd.DataFrame({"delta": delta})

    elif kernel == "Connectedness":
        log_rho = theta
        rho = torch.exp(log_rho)
        # print((1 - rho) / (1 + (n_alleles - 1) * rho))
        delta = 1 - (1 - rho) / (1 + (n_alleles - 1) * rho)
        delta = pd.DataFrame({"delta": delta}, index=positions)

    elif kernel == "Jenga":
        log_rho = theta[:, 0]
        rho = torch.exp(log_rho).unsqueeze(1)
        # print((1 - rho) / (1 + (n_alleles - 1) * rho))
        log_p = theta[:, 1:]
        norm_factor = torch.logsumexp(theta[:, 1:], axis=1).unsqueeze(1)
        log_p = log_p - norm_factor
        p = torch.exp(log_p)
        delta = 1 - torch.sign(1 - rho) * torch.sqrt(
            torch.abs(1 - rho) / (1 + (1 - p) / p * rho)
        )
        delta = pd.DataFrame(delta, columns=alleles, index=positions)
        delta = delta[alphabet]

    elif kernel == "GeneralProduct":
        delta = {}
        transform = CorrCholeskyTransform()
        for position, theta_p in zip(positions, theta):
            L = transform(theta_p)
            C = L @ L.T
            delta_p = pd.DataFrame(1 - C, index=alleles, columns=alleles)
            delta[position] = delta_p[alphabet].loc[alphabet]

    return delta


def load_params(dataset, kernel, id=60):
    fdir = "output"
    suffix = "test_pred.csv.model_params.pth"
    fname = "{}.{}.{}.{}".format(dataset, id, kernel, suffix)
    fpath = join(fdir, fname)
    params = torch.load(fpath, map_location=torch.device("cpu"))
    return params


def load_decay_rates(dataset, kernel, id=60):
    params = load_params(dataset, kernel, id)
    positions = POSITIONS.get(dataset, None)
    alphabet = ALPHABET.get(dataset, None)
    try:
        theta = params["covar_module.module.theta"]
    except KeyError:
        theta = params["covar_module.theta"]
    decay_rates = theta_to_decay_rates(theta, kernel, positions, alphabet)
    return decay_rates


def get_jenga_mut_decay_rates(decay_rates):
    mut_decay_rates = {}
    alleles = decay_rates.columns
    for pos in decay_rates.index:
        pos_decay = decay_rates.loc[pos, :].values
        v1 = np.expand_dims(1 - pos_decay, 0)
        v2 = np.expand_dims(1 - pos_decay, 1)
        deltas = 1 - v1 * v2
        np.fill_diagonal(deltas, 0)
        mut_decay_rates[pos] = pd.DataFrame(
            deltas, index=alleles, columns=alleles
        )
    return mut_decay_rates
