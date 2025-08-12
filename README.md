# Learning sequence-function relationships with scalable, interpretable Gaussian processes

This repository contains the code to reproduce the analyses and figures from our publication on using Gaussian process models to infer sequence-function relationships.

For running our code on your own data, please refer to our [EpiK](https://github.com/cmarti/epik) repository.

- Zhou J., Martí-Gómez C., Petti S., McCandlish D.M. (2025). Learning sequence-function relationships with scalable, interpretable Gaussian processes. bioRxiv.

## Repository Structure

The repository is organized into the following folders:

- `data`: Contains input data from external sources required for all calculations.
- `scripts`: Contains scripts necessary to reproduce the analyses in the paper.
    - `cluster`: Includes bash scripts for submitting GPU jobs to our cluster.
    - `figures`: Contains Python scripts for generating all figures.
    - `process`: Includes Python scripts for data preprocessing and processing files in the `output` folder to create figures.
    - `scratch`: Contains older scripts used during exploratory analyses. These scripts may not align with the current repository structure and are retained for record-keeping purposes.
- `results`: Contains result files used for figure generation.
- `output`: Stores intermediate output files generated during model fitting.
- `figures`: Contains all figure panels created using the scripts.

## Requirements

### Dependencies

- [gpmap-tools](https://github.com/cmarti/gpmap-tools)
- [EpiK](https://github.com/cmarti/epik)

### Setting Up the Environment

Create a new Conda environment:

```bash
conda create -n epik python=3.8
```

Activate the environment and install dependencies:

```bash
conda activate epik

git clone https://github.com/cmarti/epik.git
cd epik
python setup.py install
cd ..

git clone https://github.com/cmarti/gpmap-tools.git
cd gpmap-tools
python setup.py install
cd ..
```

Download the repository:

```bash
git clone git@github.com:cmarti/epik_analyses.git
```

## Figures

This repository provides all the code and data required to reproduce the figures from our study. The scripts for figure generation are located in the `scripts/figures` folder. To generate the figures, run the `make_figures.sh` script:

```bash
bash make_figures.sh
```

## Computational Analysis

The `make_analysis.sh` script outlines the steps to run various scripts for reproducing the analyses. Note that some jobs or scripts depend on the completion of other jobs. 

> **_NOTE:_** We use [EpiK](https://github.com/cmarti/epik) on V100 GPUs in the Cold Spring Harbor Laboratory high-performance computing cluster. The provided scripts in `scripts/cluster` are tailored to our system and may require adaptation for use in other computing environments.
