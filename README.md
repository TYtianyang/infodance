# Accuracy, Fairness, Diversity All at Once: An Influence Function Guided Data Enhancement Approach for Recommender System

This repository contains the code and dataset for our paper: **"Accuracy, Fairness, Diversity All at Once: An Influence Function Guided Data Enhancement Approach for Recommender System"**.

## Abstract
Recommender systems play a pivotal role in curating high-quality content for users, predominantly leveraging data-driven algorithms and machine learning methodologies. However, the intrinsic data-centric nature of these systems raises critical concerns; biased datasets and algorithms can inadvertently propagate biases to end-users. Furthermore, machine learning techniques, while powerful, can overfit a user's preference, leading to a monotonous stream of content suggestions. Both the CS and IS community have well-recognized the need of fairness and diversity in recommender systems and many studies are proposed to mitigate these challenges. Yet, a tangible solution that holistically addressed all three components—accuracy, fairness, and diversity—in unison remains elusive. This paper aims to bridge this gap, introducing a novel Influence-function-guided, Fair, and Diverse Data Enhancement (InFoDance) approach that enhances all three perspectives simultaneously.  It consists of four interconnected modules: model training, candidate data generation, influence function-based candidate evaluation, and virtual data selection. It iteratively generates virtual data to update the trained recommender system. The empirical evaluation has shown that our approach can improve accuracy, fairness, and diversity by up to 24.27%, 55.29%, and 1.85% simultaneously and significantly outperform the state-of-the-art baselines on multiple evaluation metrics. 

![image](https://github.com/user-attachments/assets/60423b26-4d79-4f7d-9894-56d1f6dcaae1)

## Download Datasets
Please first download the datasets in the corresponding folders.
- [[Movielens 1M](https://grouplens.org/datasets/movielens/1m/)] (`data/ml-1m/`);
- [[LastFM 1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)] (`data/lastfm-dataset-1K/`) .

## Preprocess Datasets
```bash
python source/preprocess.py
```

## Run Main Experiments
Run the main experiments and create result tables.
```bash
python 1_main_run.py
```
Compose aggregated result tables.
```bash
python 1_main_summary.py
```

## Visualize Trajectories & Conduct Demographic-wise Studies
Collet results.
```bash
python 2_inference_run.py
```
Visualization and aggregate results.
```bash
python 2_inference_summary.py
```
Some post processing.
```bash
python 2_inference_misc.py
```

## Motivate Studies By Introducing Sparsity
```bash
python 3_motivation.py
```

## Run Baselines
Run the main experiments for baseline methods.
```bash
python 4_baseline.py
```
Aggregate results.
```bash
python 4_baseline_summary.py
```


