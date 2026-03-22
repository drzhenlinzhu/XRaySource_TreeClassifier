# XRaySource_TreeClassifier

This notebook (`tree_based_model.ipynb`) benchmarks tree-based supervised classifiers on astrophysical X-ray sources from NASA's Chandra Source Catalog (CSC).

---

## Motivation

A labeled dataset of 8,271 uniquely classified X-ray sources is used to train and evaluate supervised tree-based classifiers, comparing Decision Tree, Random Forest, XGBoost, and LightGBM on a 4-class astrophysical source classification problem.

---

## Dataset

**Source file**: `out_data/uniquely_classified.csv`

| Property | Value |
|---|---|
| Total sources | 8,271 |
| Train / test split | 70% / 30% (stratified) |
| Number of classes | 4 (aggregated from 10 original types) |
| Random seed | 1 |

**Target variable**: `agg_master_class` — four aggregated source classes:
- Class 0: X-ray Binaries (XB)
- Class 1: AGN-type (Seyfert)
- Class 2: Young Stellar Objects (YSO)
- Class 3: AGN

**Features** (13 photometric/variability properties):

| Feature | Description |
|---|---|
| `hard_hm`, `hard_hs`, `hard_ms` | Spectral hardness ratios (hard/medium/soft bands) |
| `powlaw_gamma` | Power-law photon index |
| `bb_kt` | Black body temperature |
| `var_prob_b/h/s` | Variability probability (broad / hard / soft band) |
| `var_ratio_b/h/s` | Variability ratio (broad / hard / soft band) |
| `var_newq_b` | Normalized Q variability statistic (broad band) |
| `detection_count` | Number of detections per source |

---

## Models

### Decision Tree
Baseline single-tree classifier. Class weights applied (`{0: 0.17, 1: 0.83}`) to address class imbalance.

### Random Forest + GridSearchCV
Ensemble of decision trees with hyperparameter tuning:
- `criterion`: gini, entropy
- `min_samples_leaf`: 1, 3, 5
- `n_estimators`: 100, 300, 500
- 5-fold cross-validation, scoring: F1-macro

### XGBoost
Gradient boosted trees with fixed hyperparameters:
- `max_depth=6`, `learning_rate=0.1`, `n_estimators=300`
- `subsample=0.8`, `colsample_bytree=0.8`

### LightGBM + RandomizedSearchCV
Microsoft's LightGBM with 60-iteration random search over:
- `num_leaves`, `max_depth`, `min_child_samples`, `min_child_weight`
- `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `learning_rate`
- 5-fold CV, early stopping on validation set

---

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Decision Tree | 68% | Overfits training data (100% train accuracy) |
| Random Forest (baseline) | 76.2% | Strong improvement over single tree |
| Random Forest (tuned) | 76.6% | Marginal gain from GridSearchCV |
| XGBoost | **78%** | Best accuracy, ROC-AUC macro = 0.94 |
| LightGBM | **78%** | Tied with XGBoost; better regularization control |

**Per-class F1 scores (XGBoost)**:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Class 0 | 0.75 | 0.82 | 0.79 |
| Class 1 | 0.72 | 0.58 | 0.64 |
| Class 2 | 0.76 | 0.76 | 0.76 |
| Class 3 | 0.84 | 0.85 | 0.85 |

**XGBoost ROC-AUC**: Macro = 0.9401 | Micro = 0.9449
**Precision-Recall AP**: Mean = 0.998 across all classes

---

## Feature Importance

Top features by LightGBM gain-based importance:

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `powlaw_gamma` | 58,063 | Spectral slope — strongest discriminator |
| 2 | `hard_hs` | 28,067 | Hard-to-soft hardness ratio |
| 3 | `hard_ms` | 22,765 | Medium-to-soft hardness ratio |
| 4 | `bb_kt` | 17,835 | Black body temperature |
| 5 | `hard_hm` | 16,077 | Hard-to-medium hardness ratio |
| 6 | `var_prob_b` | 15,005 | Variability probability (broad band) |
| 7 | `var_prob_h` | 13,710 | Variability probability (hard band) |
| 8 | `var_ratio_b` | 12,275 | Variability ratio (broad band) |
| 13 | `detection_count` | 1,553 | Least informative feature |

**Key insight**: Spectral features (`powlaw_gamma`, hardness ratios) dominate, confirming that X-ray spectral shape is the primary physical discriminator between source classes. Variability features contribute but are secondary.

---

## Visualizations

- Feature importance bar plots (Decision Tree, Random Forest, LightGBM)
- Decision tree structure plot (depth ≤ 4)
- Confusion matrix heatmaps for all models
- Multiclass precision-recall curves (XGBoost, one-vs-rest)
- Decision threshold tuning analysis

---

## Key Takeaways

1. **XGBoost and LightGBM achieve 78% accuracy** — the best among all models tested, with excellent ROC-AUC (0.94) and near-perfect precision-recall AP (0.998).
2. **Spectral features dominate**: `powlaw_gamma` and hardness ratios are the most predictive features, consistent with astrophysical intuition.
3. **Class 1 (Seyfert) is hardest to classify** — lowest F1 across all models, likely due to smaller sample size and overlapping spectral properties.
4. **Hyperparameter tuning yields diminishing returns** — baseline configurations were already near-optimal; gains from grid/random search were under 1%.
5. **High AUC and AP scores** suggest the models can generalize classification to new sources with strong confidence.

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```
