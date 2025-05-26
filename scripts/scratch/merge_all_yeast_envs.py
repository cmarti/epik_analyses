#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker as lm


if __name__ == '__main__':
    subset = 'qtls'
    environments = [line.strip() for line in open('environments.txt')]

    dfs = []
    for env in environments:
        df = pd.read_csv('datasets/{}_{}_hq.csv'.format(subset, env)).groupby(['seq'])[['y']].mean()
        df.columns = [env]
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df.to_csv('datasets/{}_all_envs.csv'.format(subset))
