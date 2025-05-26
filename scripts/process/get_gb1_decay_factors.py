#!/usr/bin/env python
from os.path import join
from scripts.utils import load_decay_rates
from scripts.settings import GB1, RESULTSDIR


if __name__ == "__main__":
    dataset = GB1

    print('Extracting Connectedness model decay factors')
    connectedness = load_decay_rates(dataset=dataset, kernel="Connectedness")
    fpath = join(RESULTSDIR, "{}.connectedness_decay_rates.csv".format(dataset))
    connectedness.to_csv(fpath)

    print('Extracting Jenga model decay factors')
    jenga = load_decay_rates(dataset=dataset, kernel="Jenga")
    fpath = join(RESULTSDIR, "{}.jenga_decay_rates.csv".format(dataset))
    jenga.to_csv(fpath)
