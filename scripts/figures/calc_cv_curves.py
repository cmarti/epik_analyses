#!/usr/bin/env python
import numpy as np
import pandas as pd

from os.path import exists
from scipy.stats import pearsonr

from scripts.figures.settings import MODELS


if __name__ == "__main__":
    datasets = ["smn1", "gb1", "aav", "qtls_li_hq"]  # , 'aav', 'smn1', 'gb1']
    datasets = ["qtls_li_hq"]
    # datasets = ["gb1"]  # , 'aav', 'smn1', 'gb1']
    labels = {
        "Global epistasis": "global_epistasis",
        "Variance Component": "VC",
        "General Product": "GeneralProduct",
    }

    training_p = pd.read_csv("splits.csv").set_index("id")["p"].to_dict()
    n = len(training_p)

    for dataset in datasets:
        results = []
        print("Evaluating dataset: {}".format(dataset))

        # Load full datapset
        data = pd.read_csv("datasets/{}.csv".format(dataset), index_col=0)

        # Iterate over subsets and kernels
        for model in MODELS:
            print("\t{}".format(model))
            label = labels.get(model, model)
            # for i in range(1, n + 1):
            for i in range(61):
                
                # Load subset
                fpath = "output/{}.{}.{}.test_pred.csv".format(
                    dataset, i, label
                )
                if not exists(fpath):
                    continue
                pred = pd.read_csv(fpath, index_col=0)

                # Compute metrics and store
                pred = pred.join(data).dropna()
                if pred.shape[0] <= 2:
                    continue

                if "y_pred" not in pred.columns:
                    x, y = pred["coef"], pred["y"]
                else:
                    x, y = pred["y_pred"], pred["y"]

                p = training_p[i]
                r2 = pearsonr(x, y)[0] ** 2
                rmse = np.sqrt(np.mean((x - y) ** 2))

                record = {
                    "id": i,
                    "dataset": dataset,
                    "p_training": p,
                    "model": model,
                    "r2": r2,
                    "rmse": rmse,
                    "n_test": x.shape[0],
                }
                print(record)
                results.append(record)

        results = pd.DataFrame(results)
        results.to_csv("results/{}.cv_curves.csv".format(dataset))
