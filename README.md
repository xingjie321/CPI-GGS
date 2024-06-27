# CPI-GGS
A model for predicting compound-protein interactions.
## Create an environment and download the following libraries that python depends on

```
conda create -n CPI-GGS python = 3.8.0
```
### The packages that python depends on are as follows
torch = 1.12.1+cu113 

torch-geometric

torch-cluster = 1.6.0+pt112cu113  

torch-scatter = 2.0.9     

torch-sparse = 0.6.16+pt112cu113

torch-spline-conv = 1.2.1+pt112cu113 

numpy

networkx

rdkit

wheel
## Install all python dependent packages and enter the environment you created
```
conda activate CPI-GGS
```
## Pre-processing of data
```
python preprocess_data.py
```
## Start training
```
python training.py
```
