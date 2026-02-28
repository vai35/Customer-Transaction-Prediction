# 💳 Customer Transaction Prediction

> A binary classification project predicting whether a customer will make a transaction, using an ensemble of **CatBoost, XGBoost, and LightGBM** with Optuna hyperparameter tuning — achieving a final **ROC-AUC of 0.896** via stacked ensemble.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Memory Optimization](#memory-optimization)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Feature Selection](#feature-selection)
- [Ensemble Methods](#ensemble-methods)
- [Results Summary](#results-summary)
- [Key Features](#key-features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Conclusion](#conclusion)

---

## 🧩 Problem Statement

Predict whether a customer will make a transaction in the future based on **200 anonymized numerical features**. This is a classic imbalanced binary classification problem where the minority class (transaction = 1) represents only ~10% of the data, making standard accuracy a misleading metric.

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | No Transaction | Customer did not transact |
| 1 | Transaction | Customer made a transaction |

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| Total Records | 200,000 |
| Total Features | 200 anonymized variables (`var_0` to `var_199`) |
| Target Column | `target` (binary: 0 or 1) |
| Class Distribution | 179,902 (0) vs 20,098 (1) |
| Imbalance Ratio | ~90% / ~10% |
| Duplicates | 0 |
| Null Values | 0 |
| Raw Memory Usage | 306.7 MB |
| Optimized Memory | 152.8 MB |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 (No Transaction) | 179,902 | 89.95% |
| **1 (Transaction)** | **20,098** | **10.05%** |

> ⚠️ Highly imbalanced dataset — ROC-AUC is used as the primary metric, not accuracy.

---

## 🔄 Project Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        END-TO-END PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐
  │  Raw Dataset     │
  │  200K × 202 cols │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                  EXPLORATORY DATA ANALYSIS                   │
  │                                                              │
  │  • Class imbalance check (~90/10 split)                      │
  │  • Correlation heatmap (max ~0.08 → highly uncorrelated)     │
  │  • PCA (Standard + Robust scaling) → no structure found      │
  │  • t-SNE + UMAP on 10K sample → no separable clusters        │
  │                                                              │
  │  Conclusion: Need non-linear / ensemble models               │
  └────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │               MEMORY OPTIMIZATION                            │
  │                                                              │
  │  float64 (200 cols) → float32                                │
  │  int64   (1 col)    → int8                                   │
  │  306.7 MB           → 152.8 MB  (↓ 50%)                     │
  └────────────────────────────┬─────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────┐
  │            TRAIN-TEST SPLIT (stratified)                     │
  │                                                              │
  │  Train: 160,000 rows  |  Test: 40,000 rows                   │
  │  stratify=y to preserve 90/10 class ratio                    │
  └────────────────────────────┬─────────────────────────────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
       ┌────────────┐  ┌─────────────┐  ┌──────────────┐
       │  CatBoost  │  │   XGBoost   │  │   LightGBM   │
       │  Base Model│  │  Base Model │  │  Base Model  │
       └──────┬─────┘  └──────┬──────┘  └──────┬───────┘
              │               │                 │
              └───────────────┼─────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │    OPTUNA HYPERPARAMETER TUNING    │
              │    (25 trials per model, GPU)      │
              └───────────────┬───────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │       FEATURE SELECTION            │
              │  Cross-model importance analysis   │
              │  Intersection (16) → Union (109)   │
              └───────────────┬───────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    ENSEMBLE METHODS                          │
  │                                                              │
  │  Simple Averaging  → ROC: 0.888                              │
  │  Weighted Averaging → ROC: 0.888                             │
  │  Stacking (LR meta) → ROC: 0.896  ✅ Best                   │
  └──────────────────────────────────────────────────────────────┘
```

---

## 🔍 Exploratory Data Analysis

### Key Findings

| Finding | Detail | Implication |
|---------|--------|-------------|
| Class Imbalance | 90% / 10% split | Use `stratify`, `scale_pos_weight`, ROC-AUC |
| Feature Correlation | Max ~0.08 | Features are nearly independent |
| PCA | Variance spread across all 200 components | Dimensionality reduction won't help |
| t-SNE | No visible cluster separation | No simple low-dim structure |
| UMAP | No visible cluster separation | Confirms complex non-linear boundary |

### Dimensionality Reduction Attempts

```
Standard PCA  ──▶  Variance spread evenly → No reduction possible
Robust PCA    ──▶  Same result → Not a scaling issue
t-SNE (10K)   ──▶  Classes overlap → No separation
UMAP (10K)    ──▶  Classes overlap → No separation

Conclusion: Rely on ensemble tree models + feature importance
```

---

## 💾 Memory Optimization

Custom dtype reduction function applied to all numerical columns:

| Column Type | Before | After | Memory Saved |
|------------|--------|-------|-------------|
| 200 feature columns | float64 | float32 | ~50% |
| target column | int64 | int8 | ~87.5% |
| **Total** | **306.7 MB** | **152.8 MB** | **↓ 50.2%** |

> 💡 float16 was tested but caused standard deviation to drop to zero due to precision loss. float32 was the optimal choice — avg std deviation drop of only **0.000055**.

---

## 🏗️ Model Building

### Base Model Configuration

| Model | Key Parameters | Class Imbalance Handling |
|-------|---------------|--------------------------|
| CatBoost | iterations=1000, lr=0.05, depth=8 | `auto_class_weights='Balanced'` |
| XGBoost | n_estimators=1000, max_depth=6 | `scale_pos_weight` = 8.95 |
| LightGBM | n_estimators=1000, max_depth=-1 | `class_weight='balanced'` |

### Base Model Results (All 200 Features)

| Model | Accuracy | Precision (class 1) | Recall (class 1) | ROC-AUC |
|-------|----------|--------------------|--------------------|---------|
| CatBoost | 88% | 0.43 | 0.69 | ~0.890 |
| XGBoost | ~88% | ~0.43 | ~0.67 | ~0.888 |
| LightGBM | ~88% | ~0.43 | ~0.68 | ~0.889 |

---

## ⚙️ Hyperparameter Tuning (Optuna)

Bayesian optimization with **25 trials per model**, using GPU acceleration where available.

### Tuning Search Space

#### CatBoost
| Parameter | Search Range |
|-----------|-------------|
| `learning_rate` | 0.03 – 0.15 (log) |
| `depth` | 6 – 10 |
| `l2_leaf_reg` | 0.1 – 5.0 (log) |
| `random_strength` | 0.5 – 3.0 (log) |
| `border_count` | 64 – 254 |

#### XGBoost
| Parameter | Search Range |
|-----------|-------------|
| `n_estimators` | 500 – 2000 |
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.2 (log) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `gamma` | 0 – 10 |

#### LightGBM
| Parameter | Search Range |
|-----------|-------------|
| `n_estimators` | 500 – 2000 |
| `num_leaves` | 31 – 256 |
| `learning_rate` | 0.01 – 0.2 (log) |
| `min_child_samples` | 10 – 100 |
| `subsample` | 0.6 – 1.0 |

### Tuning Results

| Model | Base ROC-AUC | Tuned ROC-AUC | Improvement | Decision |
|-------|-------------|---------------|-------------|---------|
| CatBoost | ~0.890 | 0.888 | ↓ Worse | Use base model |
| **XGBoost** | ~0.888 | **0.894** | ↑ +0.006 | ✅ Use tuned |
| **LightGBM** | ~0.889 | **0.893** | ↑ +0.004 | ✅ Use tuned |

### Best Hyperparameters

**XGBoost (Best):**
```python
{
  'n_estimators': 2000,
  'max_depth': 3,
  'learning_rate': 0.03885,
  'subsample': 0.6025,
  'colsample_bytree': 0.8796,
  'gamma': 2.639,
  'min_child_weight': 6.925
}
```

**LightGBM (Best):**
```python
{
  'n_estimators': 1989,
  'learning_rate': 0.01523,
  'num_leaves': 44,
  'min_child_samples': 81,
  'subsample': 0.7710,
  'colsample_bytree': 0.9977
}
```

---

## 🎯 Feature Selection

### Strategy: Cross-Model Importance Analysis

```
Step 1: Get top-N important features from each model
           CatBoost top-N  ──┐
           XGBoost top-N   ──┼──▶  Set operation
           LightGBM top-N  ──┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               Intersection           Union
              (16 features)        (58–109 features)
                    │                     │
                    ▼                     ▼
             ROC drops             ROC maintained
             (too few)             (matches full 200)
```

### Feature Set Comparison

| Feature Set | # Features | XGBoost ROC | LightGBM ROC | CatBoost ROC |
|------------|------------|-------------|--------------|--------------|
| All features | 200 | 0.894 | 0.893 | 0.890 |
| Intersection (16) | 16 | ~0.82 | ~0.82 | ~0.82 |
| Union top-50 | 58 | ~0.86 | ~0.86 | ~0.86 |
| Union top-75 | 82 | ~0.88 | ~0.88 | ~0.88 |
| **Union top-100** | **109** | **~0.894** | **~0.893** | **~0.890** |

> ✅ **109 features** (union of top-100 per model) matched full-feature performance, selected for ensemble training.

### 16 Universally Important Features (Intersection)

```
var_99, var_78, var_166, var_146, var_22, var_6,
var_21, var_110, var_133, var_174, var_76, var_13,
var_190, var_53, var_109, var_12
```

---

## 🤝 Ensemble Methods

### Architecture

```
         x_train (109 features)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 CatBoost    XGBoost    LightGBM
    │           │           │
    └─────┬─────┴─────┬─────┘
          │           │
   OOF Predictions  Test Predictions
   (5-Fold CV)       (averaged)
          │
          ▼
   Meta Features (N × 3 matrix)
          │
          ▼
   Logistic Regression
   (Meta Model)
          │
          ▼
   Final Predictions
```

### Ensemble Results Comparison

| Method | ROC-AUC | Accuracy | Notes |
|--------|---------|----------|-------|
| Simple Averaging | 0.888 | — | Equal weights |
| Weighted Averaging | 0.888 | — | CatBoost 0.5, XGB 0.35, LGB 0.4 |
| **Stacking (LR Meta, 109 features)** | **0.889** | **92%** | ✅ Best balance |
| **Stacking (LR Meta, 200 features)** | **0.896** | **92%** | 🏆 **Final Best** |

### Final Model Classification Report (Stacking, 200 features)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (No Transaction) | 0.94 | 0.98 | 0.96 | 35,980 |
| 1 (Transaction) | 0.69 | 0.39 | 0.50 | 4,020 |
| **Accuracy** | | | **0.92** | **40,000** |
| Macro Avg | 0.81 | 0.69 | 0.73 | 40,000 |

---

## 📊 Results Summary

| Model | Features | ROC-AUC | Accuracy | Notes |
|-------|----------|---------|----------|-------|
| CatBoost (base) | 200 | 0.890 | 88% | Baseline |
| XGBoost (base) | 200 | 0.888 | 88% | Baseline |
| LightGBM (base) | 200 | 0.889 | 88% | Baseline |
| CatBoost (tuned) | 200 | 0.888 | — | Tuning didn't help |
| XGBoost (tuned) | 200 | 0.894 | 89% | ↑ improved |
| LightGBM (tuned) | 200 | 0.893 | — | ↑ improved |
| Simple Avg Ensemble | 109 | 0.888 | — | No gain |
| Weighted Avg Ensemble | 109 | 0.888 | — | No gain |
| Stacking (109 features) | 109 | 0.889 | 92% | Slight gain |
| **Stacking (200 features)** | **200** | **0.896** | **92%** | 🏆 **Best** |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/vai35/customer-transaction-prediction.git
cd customer-transaction-prediction

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost lightgbm optuna umap-learn torch
```

### Requirements

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation & memory optimization |
| `scikit-learn` | Preprocessing, splitting, metrics, Logistic Regression |
| `catboost` | CatBoost gradient boosting |
| `xgboost` | XGBoost gradient boosting |
| `lightgbm` | LightGBM gradient boosting |
| `optuna` | Bayesian hyperparameter optimization |
| `umap-learn` | UMAP dimensionality reduction |
| `matplotlib`, `seaborn` | Visualization |
| `torch` | GPU availability check |

---

## 📁 Project Structure

```
customer-transaction-prediction/
│
├── 📓 custTransPred.ipynb            # Main notebook
├── 📄 README.md                      # Project documentation
│
└── 📂 dataset/
    └── Data/
        └── train.csv                 # Training data (200K rows)
```

---

## 🔑 Key Features (16 Universal)

These features were identified as important across **all three models** (intersection of top-30 each):

```
var_99  | var_78  | var_166 | var_146 | var_22  | var_6
var_21  | var_110 | var_133 | var_174 | var_76  | var_13
var_190 | var_53  | var_109 | var_12
```

> Although these 16 alone weren't sufficient for best performance, they represent the **strongest predictive signals** in the dataset.

---

## 🔮 Future Scope

- 🔍 **Feature engineering** — Explore interactions between top features
- 📊 **SHAP analysis** — Deep dive into individual prediction explanations
- ⚖️ **SMOTE / oversampling** — Experiment with synthetic minority oversampling
- 🚀 **Model deployment** — Serve predictions via FastAPI
- 🔁 **AutoML comparison** — Benchmark against AutoGluon or H2O

---

## 👤 Author

**[Vaishnavi Shidling]**
- 🔗 LinkedIn: [linkedin.com/in/vaishnavi-shidling/]
- 💻 GitHub: [https://github.com/vai35/]
- 📧 Email: [vaishnavishidling74@gmail.com]

---

*Built as part of the DataMites Capstone Project — PRCP-1003*
