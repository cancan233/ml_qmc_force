# Machine Learning Diffusion Monte Carlo Forces

In this repository, we provide all necessary files to reproduce our work in "Machine Learning Diffusion Monte Carlo Forces". 

## Environment Setup

Necessary software:

- QMCPACK
- QuantumESPRESSO
- PySCF
- AMPtorch

For python environment, we follow the setup tutorial [here](https://github.com/ulissigroup/amptorch/blob/master/README.md).

## Training and Test Dataset

Due to the file size limit by Github, an extra compressed `.zip` file including all training and test dataset are provided in below link. 

https://drive.google.com/file/d/1NJuZSZlur-7os5HtXxOKgpmXuZxWtqjF/view?usp=sharing

## Scripts

The repository is organized based on the examples we studied in our work. You can find the example-related python scripts in each example subfolder. 

For example,

```
- C2
  - data: scripts to generate training, validation, and test data
  - train: scripts to train different AMPtorch models
  - md_start_eq: scripts for MD simulations using different trained models
  - geom_opt: scripts for geometry optimization using different trained models
  - notebooks_analysis: scripts for data analysis and all figures used in our work
```