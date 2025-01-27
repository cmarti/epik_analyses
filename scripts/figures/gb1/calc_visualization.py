#!/usr/bin/env python
import pandas as pd

from gpmap.src.space import SequenceSpace
from gpmap.src.randwalk import WMWalk


if __name__ == "__main__":
    dataset = "gb1"
    mean_function = 0.0
    kernel_label = "Jenga"
    id = ""

    print(
        "Generating visualization for inferred landscape {} with {} kernel".format(
            dataset, kernel_label
        )
    )

    fpath = "output/{}{}.{}.test_pred.csv".format(dataset, id, kernel_label)
    landscape = pd.read_csv(fpath, index_col=0)

    space = SequenceSpace(X=landscape.index.values, y=landscape["coef"].values)
    rw = WMWalk(space)
    rw.calc_visualization(mean_function=mean_function)
    rw.write_tables(
        "output/{}.{}".format(dataset, kernel_label),
        write_edges=True,
        nodes_format="csv",
    )
