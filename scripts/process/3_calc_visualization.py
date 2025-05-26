#!/usr/bin/env python
import pandas as pd

from gpmap.src.space import SequenceSpace
from gpmap.src.randwalk import WMWalk


if __name__ == "__main__":
    mean_function = 0.0
    fpath = "output_new/gb1.full.1.Jenga.full_pred.csv"
    landscape = pd.read_csv(fpath, index_col=0)

    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=mean_function, n_components=20)
    rw.write_tables("output_new/gb1.jenga", write_edges=True, nodes_format="csv")
