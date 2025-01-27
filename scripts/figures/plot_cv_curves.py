import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    labels = {
        "r2": "Test $R^2$",
        "rmse": "rmse",
        "logit_r2": r"$\log_2\left(\frac{V_{model}}{V_{res}}\right)$",
        "log_likelihood": "Test log(L)",
    }
    datasets = ["smn1", "gb1", "aav", "qtls_li_hq"]
    models = [
        "Additive",
        "Global epistasis",
        "Pairwise",
        "Exponential",
        "Variance Component",
        "Connectedness",
        "Jenga",
        "GeneralProduct",
    ]
    colors = [
        "silver",
        "salmon",
        "gray",
        "violet",
        "slateblue",
        "purple",
        "black",
        "gold",
    ]
    palette = dict(zip(models, colors))

    # order = ['Additive', 'Pairwise', 'Exponential', 'Connectedness', 'Jenga']
    order = models  # [:-1]
    print(order)
    metric = "r2"
    ylabel = labels[metric]
    lw = 0.6

    for dataset in datasets:
        fpath = "results/{}.cv_curves.csv".format(dataset)
        data = pd.read_csv(fpath, index_col=0)
        data["logit_r2"] = np.log2(data["r2"] / (1 - data["r2"]))
        # data = data.loc[data["r2"] > 0.2, :]

        print(
            "\tPlotting {} curve for dataset: {}".format(
                metric.upper(), dataset
            )
        )
        fig, axes = plt.subplots(1, 1, figsize=(5.0, 3))
        obs = data["model"].unique()
        hue_order = [x for x in order if x in obs]
        sns.lineplot(
            x="p_training",
            y=metric,
            hue="model",
            hue_order=hue_order,
            lw=lw,
            palette=palette,
            data=data,
            ax=axes,
            err_style="bars",
            err_kws={"capsize": 1.5, "capthick": lw, "lw": lw},
            errorbar="sd",
        )
        axes.grid(alpha=0.2)

        axes.legend(loc=(1.02, 0.2), fontsize=8, frameon=True, ncol=1)
        axes.set(
            xlabel="Proportion of training data",
            ylabel=ylabel,
            ylim=(0, 1) if metric == "r2" else (None, None),
            xlim=(0, 1),
            #  xscale='log',
            #  yscale='logit',
        )
        fig.tight_layout()
        fig.savefig("figures/{}.{}.svg".format(dataset, metric), dpi=300)
        fig.savefig("figures/{}.{}.png".format(dataset, metric), dpi=300)

    print("Done")
