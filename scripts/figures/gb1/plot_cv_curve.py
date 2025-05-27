import matplotlib.pyplot as plt

from scripts.utils import load_r2
from scripts.settings import GB1
from scripts.figures.plot_utils import plot_cv_curve, FIG_WIDTH, savefig


if __name__ == "__main__":
    dataset = GB1
    metric = "r2"
    p = 0.4

    data = load_r2(dataset)

    print("Plotting R2 curves across models")
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * p, FIG_WIDTH * p))
    plot_cv_curve(axes, data, metric=metric)
    fig.tight_layout()
    savefig(fig, "{}.{}".format(dataset, metric))
