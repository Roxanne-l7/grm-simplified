# GRM Simplified (Generative Recommendation Model)

This is a **simplified PyTorch reproduction** of the paper  
**[Generative Recommendation: Towards Next-generation Recommender Systems (NeurIPS 2023)](https://arxiv.org/abs/2305.05065)**  
implemented and tested on the **MovieLens-1M dataset**.

Unlike traditional retrieval or ranking models, GRM frames recommendation as a **generative sequence modeling problem**, where the model predicts the next item in a user’s history **as token generation** with a Transformer encoder.

---

## ✨ Highlights
- **Lightweight Reproduction**: Minimal PyTorch code (~200 lines) for easy understanding.
- **Kaggle-Friendly**: Runs directly on Kaggle Notebook with GPU.
- **Competitive Results**: Achieves Hit@10 ≈ 0.17 and NDCG@10 ≈ 0.09 on a sampled MovieLens-1M dataset.
- **Modular Code**: Clear separation of `dataset.py`, `model.py`, `train.py`, `evaluate.py`.

---

## 📂 Project Structure
- `dataset.py` → Dataset preparation (sequence truncation, padding, target building)
- `model.py` → GRM model (Embedding + Transformer Encoder + Prediction Layer)
- `train.py` → Training loop (Adam optimizer + CrossEntropy loss)
- `evaluate.py` → Evaluation metrics (Hit@K, NDCG@K)
- `requirements.txt` → Dependencies
- `README.md` → Project introduction

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run Training
```bash
python train.py
```
### 3. Evaluate
```python
from train import train
from evaluate import evaluate

model, dataloader = train()
evaluate(model, dataloader)
```

## Example Results
On a subsampled MovieLens-1M dataset (2,000 users):

Metric	Score
Hit@10	~0.17
NDCG@10	~0.09

## Reference
Paper:https://arxiv.org/abs/2305.05065
Dataset:https://grouplens.org/datasets/movielens/1m/

## Acknowledgements
This repo is a lightweight and educational reproduction, designed for learning and fast experimentation.
For full-scale implementations, please refer to official research repos.
