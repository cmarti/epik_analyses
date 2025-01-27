#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
import logomaker

    

if __name__ == '__main__':
    data = pd.read_csv('datasets/aav.csv', index_col=0)
    
    m = logomaker.alignment_to_matrix(data.index.values, to_type='probability', pseudocount=0)
    m.index = np.arange(561, 561 + 28)
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 1.5))
    
    logomaker.Logo(m, ax=axes, color_scheme='chemistry')
    axes.set(xlabel='Position', ylabel='Frequency')
        
    fig.tight_layout()
    fig.savefig('plots/aav_data_distrib.png', dpi=300)
    