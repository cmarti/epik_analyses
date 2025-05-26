#!/usr/bin/env python
import numpy as np
import pandas as pd


from gpmap.seq import generate_possible_sequences, guess_space_configuration
from gpmap.linop import calc_covariance_vjs


if __name__ == "__main__":
    for dataset in ["smn1", "gb1"]:
        # Load data
        fpath = "datasets/{}.csv".format(dataset)
        data = pd.read_csv(fpath, index_col=0)
        config = guess_space_configuration(
            data.index.values, ensure_full_space=False
        )
        print(config)
        a, l, alphabet = (
            config["n_alleles"][0],
            config["length"],
            np.unique(np.hstack(config["alphabet"])),
        )

        X = np.array(
            list(generate_possible_sequences(seq_length=l, alphabet=alphabet))
        )
        idx = pd.Series(np.arange(X.shape[0]), index=X)
        obs_idx = idx.loc[data.index.values].values
        y = data.y.values
        cov, ns, sites_matrix = calc_covariance_vjs(y - y.mean(), a, l, obs_idx)
        alleles = ["A", "B"]
        seqs = np.array(
            ["".join([alleles[i] for i in x]) for x in sites_matrix.astype(int)]
        )

        with open("datasets/{}.seqs.txt".format(dataset), "w") as fhand:
            for seq in X:
                fhand.write("{}\n".format(seq))

        # # Load model params
        # fpath = "output/{}.Rho.test_pred.csv.model_params.pth".format(dataset)
        # params = torch.load(fpath, map_location=torch.device("cpu"))
        # logit_rho = (
        #     params["covar_module.logit_rho"].detach().cpu().numpy().flatten()
        # )
        # rho = np.exp(logit_rho) / (1 + np.exp(logit_rho))
        # prior_cov = [
        #     np.prod(
        #         [
        #             1 + (a - 1) * r if i == 0 else 1 - r
        #             for i, r in zip(sites, rho)
        #         ]
        #     )
        #     for sites in sites_matrix
        # ]

        # # Load MAP
        # fpath = "output/{}.Rho.test_pred.csv".format(dataset)
        # map = pd.read_csv(fpath, index_col=0)
        # y = map.y_pred.values
        # cov_map, ns_map, _ = calc_covariance_vjs(y - y.mean(), a, l)

        covs = pd.DataFrame(
            {
                "d": [x.count("B") for x in seqs],
                "data": cov / cov[0],
                "data_ns": ns,
                # "prior": prior_cov / prior_cov[0],
                # "map": cov_map / cov_map[0],
                # "map_ns": ns_map,
            },
            index=seqs,
        )
        covs.to_csv("{}.vj_covariances.csv".format(dataset))
