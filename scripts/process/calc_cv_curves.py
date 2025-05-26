#!/usr/bin/env python
import numpy as np
import pandas as pd

from os.path import exists, join
from scipy.stats import pearsonr

from scripts.settings import (
    MODELS,
    MODEL_KEYS,
    DATASETS,
    DATADIR,
    SPLITSDIR,
    PARAMSDIR,
    RESULTSDIR,
)


def load_data_pred(dataset, i, model):
    label = MODEL_KEYS.get(model, model)
    fname = "{}.{}.{}.test_pred.csv".format(dataset, i, label)
    fpath = join(PARAMSDIR, fname)

    if exists(fpath):
        return pd.read_csv(fpath, index_col=0)
    return None


def load_mll(dataset, i, model):
    label = MODEL_KEYS.get(model, model)
    fname = "{}.{}.{}.loss.csv".format(dataset, i, label)
    fpath = join(PARAMSDIR, fname)
    if exists(fpath):
        mll = pd.read_csv(fpath, index_col=0)["mll"].values[-10:].mean()
    else:
        mll = np.nan
    return mll


def get_n_training(dataset, i):
    fpath = join(SPLITSDIR, "{}.{}.train.csv".format(dataset, i))
    n_train = pd.read_csv(fpath, index_col=0).shape[0]
    return n_train


def evaluate_predictions(pred, data):
    pred = pred.join(data).dropna()
    if pred.shape[0] <= 2:
        raise ValueError(
            "Not enough data points for evaluation: {}".format(pred.shape[0])
        )

    if "y_pred" not in pred.columns:
        x, y = pred["coef"], pred["y"]
    else:
        x, y = pred["y_pred"], pred["y"]

    r2 = pearsonr(x, y)[0] ** 2
    rmse = np.sqrt(np.mean((x - y) ** 2))
    n_test = x.shape[0]
    return (r2, rmse, n_test)


if __name__ == "__main__":
    fpath = join(DATADIR, "splits.csv")
    training_p = pd.read_csv(fpath).set_index("id")["p"].to_dict()
    n = len(training_p)

    for dataset in DATASETS:
        results = []
        print("Evaluating dataset: {}".format(dataset))

        # Load full datapset
        fpath = join(DATADIR, "{}.csv".format(dataset))
        data = pd.read_csv(fpath, index_col=0)

        # Iterate over subsets and kernels
        for model in MODELS:
            print("\t{}".format(model))

            for i, p in training_p.items():
                pred = load_data_pred(dataset, i, model)
                if pred is None:
                    continue

                n_train = get_n_training(dataset, i)
                mll = load_mll(dataset, i, model)
                r2, rmse, n_test = evaluate_predictions(pred, data)

                record = {
                    "id": i,
                    "dataset": dataset,
                    "p_training": p,
                    "model": model,
                    "r2": r2,
                    "rmse": rmse,
                    "n_test": n_test,
                    "mll": mll,
                    "n_train": n_train,
                }
                results.append(record)

        results = pd.DataFrame(results)
        fpath = join(RESULTSDIR, "{}.cv_curves.csv".format(dataset))
        results.to_csv(fpath)
