from os.path import join

import pandas as pd
import numpy as np
import torch

from gpmap.utils import read_edges
from torch.distributions.transforms import CorrCholeskyTransform
from scripts.settings import (
    ALPHABET,
    POSITIONS,
    PARAMSDIR,
    RESULTSDIR,
    GB1_PEAK_SEQS,
)


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
        log_rho = theta.flatten()
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


def torch_load(fpath):
    return torch.load(fpath, map_location=torch.device("cpu"))


def load_params(dataset, kernel):
    suffix = "model_params.pth"
    fname = "{}.full.1.{}.{}".format(dataset, kernel, suffix)
    fpath = join(PARAMSDIR, fname)

    params = torch_load(fpath)
    return params


def params_to_decay_rates(params, dataset, kernel):
    positions = POSITIONS.get(dataset, None)
    alphabet = ALPHABET.get(dataset, None)
    try:
        theta = params["covar_module.module.theta"]
    except KeyError:
        theta = params["covar_module.theta"]
    decay_rates = theta_to_decay_rates(theta, kernel, positions, alphabet)
    return decay_rates


def load_decay_rates(dataset, kernel):
    params = load_params(dataset, kernel)
    return params_to_decay_rates(params, dataset, kernel)


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


def load_gb1_visualization():
    print("Loading correlation with {} under all models".format(GB1_PEAK_SEQS))
    fpath = join(RESULTSDIR, "gb1.kernels_at_peaks.csv")
    kernel = pd.read_csv(fpath, index_col=0)

    print("Loading visualization coordinates")
    fpath = join(RESULTSDIR, "gb1.jenga.nodes.csv")
    nodes = pd.read_csv(fpath, index_col=0)
    edges = read_edges(fpath=join(RESULTSDIR, "gb1.jenga.edges.npz"))
    nodes = nodes.join(kernel)

    return (nodes, edges)
