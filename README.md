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

## 1. Introduction
Breast cancer risk prediction can benefit from longitudinal mammograms and complementary CC and MLO views. However, existing approaches often rely on a fixed number of prior examinations and do not explicitly account for anatomical differences across time. FlexLMV-Net addresses these limitations by jointly modeling multi-view longitudinal mammograms, aligning all prior examinations to the current examination, and supporting a variable number of prior exams for flexible risk prediction. Evaluated on the EMBED and CSAW-CC datasets, FlexLMV-Net demonstrates improved breast cancer risk prediction across multiple follow-up years, breast density categories, and cancer subgroups. These results highlight the potential of flexible longitudinal multi-view modeling for improved risk stratification and personalized breast cancer screening.


## 2. Method
FlexLMV-Net is a longitudinal multi-view deep learning model for five-year breast cancer risk prediction from screening mammograms. It builds upon our previously published LMV-Net (https://github.com/sot176/LMV-Net) and extends it to jointly model current and prior CC/MLO views while supporting a variable number of prior examinations, including first-time attendees with no prior examinations.


### Key components:

**1) Explicit longitudinal multi-view alignment:** All prior mammograms are aligned to the current examination using pretrained MammoRegNet, with aligned prior and sequential difference features adaptively fused using spatial attention and time-aware feature weighting to support a variable number of prior examinations.

**2) Multi-View Dual-stream attention:** Self- and cross-view attention jointly capture within-view and complementary CC/MLO information.

**3) Risk prediction Head:** Fused multi-view representations are used to predict five-year breast cancer risk with a cumulative probability formulation.


<p align="center">
  <img src="Figures/FlexLMV-Net.png" alt="Image 1" width="500"/>
</p>

## 3. Datasets
We used two large, publicly available mammography datasets :
- **Emory Breast Imaging Dataset (EMBED)**: https://aws.amazon.com/marketplace/pp/prodview-unw4li5rkivs2#overview}
- **Cohort of Screen-Aged Women Case Control (CSAW-CC)**: https://snd.se/en/catalogue/dataset/2021-204-1

## 4. Key findings of the paper

❌ **Existing Gap:** Current breast cancer risk prediction models do not effectively integrate complementary information from multiple mammographic views and longitudinal examinations, despite both being routinely available in screening.

🛠️ **Proposed Solution:** FlexLMV-Net jointly models multi-view and longitudinal mammograms while supporting a variable number of prior examinations, enabling flexible use of available patient history.

🚀 **Novelty:** Combines explicit temporal feature alignment, feature-aware temporal fusion, dual-stream attention, and multi-view learning to capture longitudinal breast changes and complementary CC/MLO information.

📊 **Robust Performance:** Demonstrates consistent improvements across breast density categories and cancer subgroups, including invasive and non-invasive cancers.

🏆 **Outperforms SOTA:** FlexLMV-Net consistently outperforms state-of-the-art approaches, including Mirai, VMRA-MaR, OA-BreaCR, and ImgFeatAlign, across C-index and AUC metrics over multiple follow-up years.


## 5. Results
#### Comparison with state of the art methods: 



#### Performance across density categories and cancer subgroups



##  6. Reproduction of the results
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


## 7. Citation
```bibtex
@inproceedings{flexlmv_net_2026,
author = {Thrun, Solveig and Sun, Zijun and  Salahuddin, Suaiba A. and Wickstrøm, Kristoffer and Wetzer, Elisabeth and Hansen, Stine, and Jenssen, Robert and Kampffmeyer, Michael},
title={Explicitly Aligned Multi-View Breast Cancer Risk Prediction from Variable-Length Screening Histories},
}
```