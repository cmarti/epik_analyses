### Description

This repository contains code for generating R2 curves for a range of training proportions across different datasets for evaluating different kernels for gaussian process regression on sequence-function releationships


### Requirements

- [gpmap-tools](https://github.com/cmarti/gpmap-tools)
- [epik](https://github.com/cmarti/epik)

Create a new environment

```bash
conda create -n epik python=3.8
```

Download and install dependencies

```bash
git clone git@github.com:cmarti/epik.git
cd epik
python setup.py install
cd ..

git clone git@github.com:cmarti/gpmap_tools.git
cd gpmap_tools
python setup.py install
cd ..
```

Download repository

```
git clone git@github.com:cmarti/epik_analyses.git
```


### Datasets

The datasets included can be found in the folder `datasets`. Each consists of a CSV file with the sequence and their associated measurement with the uncertainty. The scripts will only compute those that are not commented out in the file `datasets.txt`, which can be sued to compute the CV curves on subsets of the datasets

### Scripts

The first script splits each of the dataset into training and validation sets with different proportions to evaluate the importance of the amount of data for prediction and whether performance saturates at a given concentration of data or continuously increases. 

```bash
bash 1_split.sh
```

The second script submits independent jobs using SGE for fitting Gaussian process regression models under different kernels to each of the generated training and validation subsets. It can be easily modified to run sequentially on a single machine. 

```bash
bash 2_fit_models.sh
```

The third script evaluates the performance of every model fit on the test data and writes a table with the R2 and other metrics for each of the training and test subsets. 

```bash
bash 3_calc_r2s.sh
```

