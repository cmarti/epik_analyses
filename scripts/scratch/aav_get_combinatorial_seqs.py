#!/usr/bin/env python
import numpy as np

from itertools import product
from scripts.figures.settings import AMINOACIDS

if __name__ == "__main__":
    wt_seq = "DEEEIRTTNPVATEQYGSVSTNLQRGNR"
    bc = "DEEEIR{}{}NP{}{}TEQYGSVSTNLQRGNR"
    labels = [x for x in product(AMINOACIDS, repeat=4)]
    seqs = [bc.format(*label) for label in labels]
    labels = ["".join(x) for x in labels]

    np.save("datasets/aav.pred.labels.npy", labels)

    with open("datasets/aav.seqs.txt", "w") as fhand:
        for seq in seqs:
            fhand.write("{}\n".format(seq))
