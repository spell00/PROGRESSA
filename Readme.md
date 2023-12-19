# PROGRESSA

This repository contains the contain to reproduce the results of the paper 
"Deep Learning-Based Algorithm to Predict Aortic Stenosis Progression from the PROGRESSA cohort"

## Install

Install the package and necessary dependencies by running the 
command: 

`pip3 install .` 

or

`python3 setup.py build` <br/>
`python3 setup.py install`

## PREPROCESS

Before training the models, the first step is to run the script to extract the features using the command:

`python3 progressa/preprocess/extract_features.py`

The next step is to create a file than will contain the indices for 
the 10 splits that will be used for the repeated holdout method. To achieve this,
run the script:

`python3 progressa/preprocess/create_splits.py`

Then, the features importance is calculated using the script :

`python3 progressa/preprocess/feature_importance.py`

The most important features returned from running this script were then entered in 
the `select_features.py` script. To get the file with selected features only, which 
will be found in `data/features-22.pkl`, run:

`python3 progressa/preprocess/select_features.py`

## Train models

To train the RNN model (GRU):

`python3 progressa/train_models/RNN.py`

To train the machine learning models compared with RNN, use the following command lines:

`python3 progressa/train_models/sklearn_models.py --model=naiveBayes`
`python3 progressa/train_models/sklearn_models.py --model=Logistic_Regression`
`python3 progressa/train_models/sklearn_models.py --model=lightgbm`

To train on 2 visits at a time, modify the precedent commands by adding the command `--n_visits=2`.
For example:

`python3 progressa/train_models/sklearn_models.py --model=naiveBayes --n_visits=2`

## Analysis

To reproduce the analysis from the paper, run the following commands

`python3 progressa/analysis/analyse_results.py`

`python3 progressa/analysis/analyse_results_per_sex.py`

`python3 progressa/analysis/severity_baseline.py`

`python3 progressa/analysis/stats.py`

`python3 progressa/analysis/calibration_plot.py`


## Create images

To reproduce the images from the paper, run the following commands

`python3 progressa/create_images/plot_rocs.py`

`python3 progressa/create_images/plot_tsne.py`



