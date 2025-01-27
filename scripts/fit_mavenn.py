#!/usr/bin/env python
import argparse
import sys
import time

import numpy as np
import pandas as pd

from os.path import exists
from mavenn import Model, load


def prep_input_data(data):
    X = data.index.values
    seq_length = len(X[0])
    alphabet = np.unique(np.vstack([[c for c in x] for x in X]))
    y = data.values[:, 0]
    if data.shape[1] > 1:
        dy = np.sqrt(data.values[:, 1])
        dy[dy < 0.0001] = 0.0001
    else:
        dy = None
    return (X, y, dy, seq_length, alphabet)


def read_test_sequences(test_fpath):
    seqs = np.array([])
    if test_fpath is not None:
        seqs = np.array([line.strip().strip('"') for line in open(test_fpath)])
    return seqs


class LogTrack(object):
    def __init__(self, fhand=None):
        if fhand is None:
            fhand = sys.stderr
        self.fhand = fhand
        self.start = time.time()

    def write(self, msg, add_time=True):
        if add_time:
            msg = "[ {} ] {}\n".format(time.ctime(), msg)
        else:
            msg += "\n"
        self.fhand.write(msg)
        self.fhand.flush()

    def finish(self):
        t = time.time() - self.start
        self.write("Finished succesfully. Time elapsed: {:.1f} s".format(t))


def main():
    print(sys.executable)
    description = (
        "Runs Mave-NN global epistasis regression on sequence space using data"
    )
    description += " from quantitative phenotypes associated to their corresponding"
    description += " sequences. If provided, the variance of the estimated "
    description += " quantitative measure can be incorporated into the model"

    # Create arguments
    parser = argparse.ArgumentParser(description=description)
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "-d", "--data", default=None, help="CSV table with genotype-phenotype data"
    )
    input_group.add_argument(
        "-m",
        "--model",
        default=None,
        help="Previously trained model to use for predictions",
    )

    options_group = parser.add_argument_group("Options")
    options_group.add_argument(
        "-n",
        "--n_iter",
        default=200,
        type=int,
        help="Number of iterations for optimization (200)",
    )
    options_group.add_argument(
        "-r",
        "--learning_rate",
        default=0.1,
        type=float,
        help="Learning rate for optimization (0.1)",
    )
    options_group.add_argument(
        "-l2",
        "--theta_regularization",
        default=None,
        type=float,
        help="L2 theta regularization (Optimize through CV)",
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("-o", "--output", required=True, help="Output file")
    output_group.add_argument(
        "-p", "--pred", help="File containing sequences for predicting genotype"
    )

    # Parse arguments
    parsed_args = parser.parse_args()
    data_fpath = parsed_args.data

    model_fpath = parsed_args.model
    n_iter = parsed_args.n_iter
    learning_rate = parsed_args.learning_rate
    theta_regularization = parsed_args.theta_regularization

    pred_fpath = parsed_args.pred
    out_fpath = parsed_args.output

    # Run
    log = LogTrack()

    log.write("Read input data from {}".format(data_fpath))
    data = pd.read_csv(data_fpath, index_col=0).dropna()

    log.write("Preparing data for modeling")
    X, y, dy, seq_length, alphabet = prep_input_data(data)

    if model_fpath is not None and exists(model_fpath):
        log.write("Load Mave-NN model from {}".format(model_fpath))
        model = load(model_fpath)
    else:
        log.write("Building Mave-NN model")
        model = Model(
            L=seq_length,
            alphabet=alphabet,
            regression_type="GE",
            gpmap_type="additive",
            ge_nonlinearity_type="nonlinear",
            ge_nonlinearity_monotonic=True,
            theta_regularization=theta_regularization,
            eta_regularization=0.01,
        )

    log.write("Setting data for model fitting")
    model.set_data(X, y, dy=dy)

    if n_iter > 0:
        log.write("Fitting the Mave-NN model")
        model.fit(
            epochs=n_iter,
            batch_size=data.shape[0],
            learning_rate=learning_rate,
            early_stopping=False,
        )

    # Write model
    fpath = "{}.model".format(out_fpath)
    log.write("Storing model at {}".format(fpath))
    model.save(fpath)

    if pred_fpath is not None:
        log.write("Reading test data to predict from {}".format(pred_fpath))
        X_test = read_test_sequences(pred_fpath)

        log.write("Obtain phenotypic predictions for test data")
        phi_test = model.x_to_phi(X_test)
        y_test = model.x_to_yhat(X_test)
        result = pd.DataFrame({"y_pred": y_test, "phi_pred": phi_test}, index=X_test)

        fpath = "{}.test_pred.csv".format(out_fpath)
        log.write("\tWriting predictions to {}".format(fpath))
        result.to_csv(fpath)

    log.finish()


if __name__ == "__main__":
    main()
