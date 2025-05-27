#!/usr/bin/env python
import numpy as np

from os.path import join
from scripts.utils import load_decay_rates, get_jenga_mut_decay_rates
from scripts.settings import SMN1, RESULTSDIR


if __name__ == "__main__":
    dataset = SMN1

    print('Extracting connectedness model decay factors')
    connectedness = load_decay_rates(dataset=dataset, kernel="Connectedness")
    fpath = join(RESULTSDIR, "{}.connectedness_decay_rates.csv".format(dataset))
    connectedness.to_csv(fpath)

    print('Extracting Jenga model decay factors')
    jenga = load_decay_rates(dataset=dataset, kernel="Jenga")
    jenga.loc["+2", ["A", "G"]] = np.nan
    fpath = join(RESULTSDIR, "{}.jenga_decay_rates.csv".format(dataset))
    jenga.to_csv(fpath)
    
    jenga = get_jenga_mut_decay_rates(jenga)
    for site, df in jenga.items():
        fname = "{}.jenga_decay_rates.{}.csv".format(dataset, site)
        df.to_csv(join(RESULTSDIR, fname))
    
    print('Extracting general product model decay factors')
    general_product = load_decay_rates(dataset=dataset, kernel="GeneralProduct")
    for site, df in general_product.items():
        if site == '+2':
            df.loc[:, ["A", "G"]] = np.nan
            df.loc[["A", "G"], :] = np.nan
        fname = "{}.general_product_decay_rates.{}.csv".format(dataset, site)
        df.to_csv(join(RESULTSDIR, fname))
