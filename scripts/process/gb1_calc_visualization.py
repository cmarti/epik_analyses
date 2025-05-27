#!/usr/bin/env python
import pandas as pd

from scripts.settings import PARAMSDIR, RESULTSDIR
from os.path import join
from gpmap.space import SequenceSpace
from gpmap.randwalk import WMWalk


if __name__ == "__main__":
    print("Loading GB1 Jenga landscape predictions")
    fpath = join(PARAMSDIR, "gb1.full.1.Jenga.full_pred.csv")
    landscape = pd.read_csv(fpath, index_col=0)

    print("Calculating visualization for GB1 Jenga landscape")
    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=0.0, n_components=10)
    
    fpath = join(RESULTSDIR, "gb1.jenga")
    print('Saving visualization coordinates at {}'.format(fpath))
    rw.write_tables(fpath, write_edges=True, nodes_format="csv")
