# Learning sequence-function relationships with scalable, interpretable Gaussian processes


This repository contains code to reproduce the analyses and figures from our 
publication on using Gaussian process models to infer sequence-function
relationships. For running our code on your own data see our [EpiK](https://github.com/cmarti/epik) repository.

- Zhou J., Martí-Gómez C., Petti S., McCandlish D.M. (2025) Learning sequence-function relationships with scalable, interpretable Gaussian processes. biorxiv

The repository contains a series of folders with different contents

- `data`: this folder contains input data from other sources required to do all calculations
- `scripts`: this folder contains all the scripts required to reproduce the analyses in the paper
  - `cluster`: this subfolder contains bash scripts to send GPU jobs in our cluster
  - `figures`: this subfolder contains python scripts to make all the figures
  - `process`: this subfolder contains python scripts to pre-process the data and process files in the `output` folder for making the figures
  - `scratch`: this subfolder contains old scripts used during the exploratory analyses and are not necessarily consistent with the current structure of the repository. They are only kept for our record. 
- `results`: this folder contains all the results files used for creating the figures
- `output`: this folder will contain the intermediate output files from fitting the models
- `figures`: this folder will contain all the figure panels generated with the scripts


### Requirements: UPDATE needed

- [gpmap-tools](https://github.com/cmarti/gpmap-tools)
- [EpiK](https://github.com/cmarti/epik)

Create a new environment

```bash
conda create -n epik python=3.8
```

Download and install dependencies

```bash
conda actitvate epik


git clone https://github.com:cmarti/epik.git
cd epik
python setup.py install
cd ..

git clone https://github.com/cmarti/gpmap-tools.git
cd gpmap_tools
python setup.py install
cd ..
```

Download repository

```
git clone git@github.com:cmarti/epik_analyses.git
```

### Figures

This repository provides all the code and data required to reproduce all the 
figures from our study in the `scripts/figures` folder as shown in `make_figures.sh`,
which can be run as follows:

```bash
bash make_figures.sh
```

### Computational analysis

The file `make_analysis.sh` shows how to run the different scripts in order to reproduce our analyses. This file is aimed to guide the analysis but some jobs
or scripts require other jobs to finish to be able to run. 

> **_NOTE:_** We run [EpiK](https://github.com/cmarti/epik) using our V100 GPUs from the Cold Spring Harbor Laboratory high performance computing cluster. The provided scripts at `scripts/cluster` are tailored to our system and may need to be adapted to run in a different computing ecosystem. 


### Yeast genotype encoding
RM = A
BY = B
RM pump is the bad one