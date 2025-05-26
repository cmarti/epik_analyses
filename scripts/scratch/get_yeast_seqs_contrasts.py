#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

    

if __name__ == '__main__':
    ena1_pos = 12
    seq_length = 83
    
    backgrounds = ['A'] * seq_length, ['B'] * seq_length

    for background in backgrounds:
        for allele in 'AB':
            seq = background.copy()
            seq[ena1_pos] = allele
            print(''.join(seq))