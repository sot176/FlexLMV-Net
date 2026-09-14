# Explicitly Aligned Multi-View Breast Cancer Risk Prediction from Variable-Length Screening Histories

This is the code for our paper "Explicitly Aligned Multi-View Breast Cancer Risk Prediction from Variable-Length Screening Histories".

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sot176/FlexLMV-Net/blob/main/LICENSE)

## Table of Contents
1. 📘 [Introduction](#introduction)  
2. ⚙️ [Method](#method)  
3. 📚 [Datasets](#datasets)  
4. 🔍 [Key findings of the paper](#key-findings-of-the-paper)  
5. 📊 [Results](#results)
6. ▶️ [Reproduction of the results](#reproduction-of-the-results)  
7. 📄 [Citation](#citation)  

## Introduction

## Method


## Datasets
We used two large, publicly available mammography datasets :
- **Emory Breast Imaging Dataset (EMBED)**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview}
- **Cohort of Screen-Aged Women Case Control (CSAW-CC)**: https://snd.se/en/catalogue/dataset/2021-204-1

## Key findings of the paper

## Results
#### Comparison with state of the art methods: 



#### Performance across density categories and cancer subgroups



##  Reproduction of the results
For reproducing the results follow the instructions below:

**Important**: for each script in the `scripts` folder, make sure you update the paths to load the correct datasets and export the results in your favorite directory.

### 1) Requirements
Requirements are in the requirements.txt file

### 2) Pre-processing of the datasets
The preprocessing step ensures that the datasets are properly prepared before training.

The `preprocessing` folder contains  the necessary scripts to preprocess images and split the datasets into training, validation and test.

To preprocess the EMBED dataset, use: `preprocessing/preprocess_img_embed.py`

To preprocess the CSAW-CC dataset, use: `preprocessing/preprocess_img_csaw_cc.py`

To split the datasets into training, validation, and test sets, use: `preprocessing/split_data.py`

Create a CSV file describing your dataset by running the notebooks in the `notebooks` folder

### 3) Training 
Run the following script `scripts/train.sh`

### 4) Inference
Run the following script `scripts/test.sh`


## Citation
```bibtex
@inproceedings{lmv_net_2026,
author = {Thrun, Solveig and Sun, Zijun and  Salahuddin, Suaiba A. and Wickstrøm, Kristoffer and Wetzer, Elisabeth and Hansen, Stine, and Jenssen, Robert and Kampffmeyer, Michael},
title={Explicitly Aligned Multi-View Breast Cancer Risk Prediction from Variable-Length Screening Histories},
}
```