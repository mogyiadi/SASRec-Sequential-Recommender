# SASRec – Assignment 2

## Overview
This project implements SASRec (Self-Attentive Sequential Recommendation) for next-item prediction using the MovieLens 1M dataset. The model is trained and evaluated using Recall@K and NDCG@K metrics.

Three model variants are included:
- A: Baseline model  
- B: Deeper model  
- C: Wider model  

## Requirements
Install dependencies:

```bash
pip install torch numpy pandas tqdm
```

## How to Run

### 0. Preprocess dataset
Before training, preprocess the MovieLens dataset:

```bash
python data_preprocessing.py
```

This generates:
- train.json
- val.json
- test.json

### 1. Train models
Train all SASRec variants (A, B, C):

```bash
python train.py
```

This script:
- trains all three model variants
- uses early stopping based on validation NDCG@10
- saves trained models to:

```
saved_models/sasrec_A.pth
saved_models/sasrec_B.pth
saved_models/sasrec_C.pth
```

### 2. Evaluate models
Run evaluation on the test set:

```bash
python evaluate.py
```

This outputs:
- Recall@10, Recall@20  
- NDCG@10, NDCG@20  

## Dataset Format
Each sample contains:
- input: sequence of item interactions
- target: next item to predict

## Notes
- Evaluation uses randomly sampled negatives, so results may vary slightly between runs.
- No fixed seed is used during evaluation.
- Model configurations are defined in `SASRec.py`.
