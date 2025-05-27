import pandas as pd
import matplotlib.pyplot as plt

from os.path import join

from scripts.settings import GB1, RESULTSDIR
from scripts.figures.plot_utils import plot_cv_curve, FIG_WIDTH, savefig


if __name__ == "__main__":
    dataset = GB1
    metric = "r2"
    fraction_width = 0.4

    print('Loading {} R2 data'.format(dataset))
    fpath = join(RESULTSDIR, "{}.cv_curves.csv".format(dataset))
    data = pd.read_csv(fpath, index_col=0)

    print('Plotting R2 curves across models')
    fig, axes = plt.subplots(
        1, 1, figsize=(FIG_WIDTH * fraction_width, FIG_WIDTH * fraction_width)
    )
    plot_cv_curve(axes, data, metric=metric)
    fig.tight_layout()
    savefig(fig, "{}.{}".format(dataset, metric))
    
