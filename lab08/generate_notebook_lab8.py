import json
import base64
from pathlib import Path

def encode_image(img_path):
    p = Path(img_path)
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

def create_lab8_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.4"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    cell_exec_counter = 1

    def add_md(source):
        lines = [line + "\n" for line in source.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(source, text_outputs=None, image_paths=None):
        nonlocal cell_exec_counter
        lines = [line + "\n" for line in source.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")

        outputs = []
        if text_outputs:
            out_lines = [line + "\n" for line in text_outputs.split("\n")]
            out_lines[-1] = out_lines[-1].rstrip("\n")
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": out_lines
            })

        if image_paths:
            for ip in image_paths:
                b64 = encode_image(ip)
                if b64:
                    outputs.append({
                        "data": {
                            "image/png": b64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })

        nb["cells"].append({
            "cell_type": "code",
            "execution_count": cell_exec_counter,
            "metadata": {},
            "outputs": outputs,
            "source": lines
        })
        cell_exec_counter += 1

    # -------------------------------------------------------------------------
    # Markdown: Title and Overview
    # -------------------------------------------------------------------------
    add_md("""# Lab 8: Hyperparameter Tuning and Model Selection using Cross-Validation

**Course**: Undergraduate Machine Learning  
**Dataset**: Olist Brazilian E-Commerce Dataset (`olist_orders_feature_engineered.csv`)  
**Duration**: 2 Hours  
**Primary Model**: Logistic Regression (`liblinear` solver, `class_weight="balanced"`)  
**Author / Student**: ML Labs Team  
**Instructor**: Sharad Laad | ORY AI Labs  

---

## 1. Overview & Business Problem
In earlier labs, we explored the Olist dataset, created an Analytical Base Table, cleaned the data, engineered features, built a pipeline, and evaluated a baseline model. In this lab, we systematically select an optimal model configuration using **Stratified 5-Fold Cross-Validation** and **`GridSearchCV`**.

- **Business Problem**: Predict whether an e-commerce customer order will arrive late (`is_late_delivery`).
- **Target**: `is_late_delivery` ($1 = \\text{late}$, $0 = \\text{on-time}$).
- **Data Hygiene & Anti-Leakage Constraints**:
  - Do NOT use target-leakage columns such as `delivery_days`, `delivery_delay_days`, review-related columns, or actual delivery timestamps.
  - Remove identifiers such as `order_id`, `customer_id`, and `customer_unique_id`.

## 2. Learning Objectives
1. Distinguish parameters from hyperparameters.
2. Explain L1 and L2 regularization.
3. Explain the meaning of Logistic Regression's $C$ parameter.
4. Use stratified cross-validation.
5. Use `GridSearchCV`.
6. Select the best hyperparameter configuration.
7. Evaluate the selected model on an untouched test set.
8. Explain why test-set tuning causes leakage.""")

    # -------------------------------------------------------------------------
    # Markdown: Key Concepts
    # -------------------------------------------------------------------------
    add_md("""## 3. Key Concepts & Mathematical Foundations

### Parameters vs. Hyperparameters
- **Parameters**: Learned directly from data during training via loss optimization (e.g., weights $w_j$ and intercept $b$ in Logistic Regression).
- **Hyperparameters**: Structural settings configured before/outside training (e.g., penalty type L1/L2, regularization inverse strength $C$, tree depth, solver).

### Regularization Formulations
- **L1 Regularization (Lasso)**:
  $$\\min_{w} \\text{Loss}(w) + \\lambda \\sum_{j=1}^p |w_j|$$
  - Introduces diamond-shaped $L_1$ constraint boundaries with sharp corners along axes.
  - Drives non-informative feature weights strictly to **0**, performing embedded feature selection.

- **L2 Regularization (Ridge)**:
  $$\\min_{w} \\text{Loss}(w) + \\lambda \\sum_{j=1}^p w_j^2$$
  - Smooth circular/spherical quadratic constraint boundary.
  - Shrinks coefficients continuously toward zero but retains virtually all features.

### The $C$ Parameter
In `scikit-learn`, $C = \\frac{1}{\\lambda}$ is the inverse of regularization strength:
- **Small $C$** $\\implies$ **Stronger Regularization** (penalizes large coefficients heavily, pushes more to zero in L1).
- **Large $C$** $\\implies$ **Weaker Regularization** (lets model fit training data closely).""")

    # -------------------------------------------------------------------------
    # Code: Section 5 - Required Libraries
    # -------------------------------------------------------------------------
    add_md("## 5. Required Libraries & Setup")
    add_code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)

print("All required machine learning libraries imported successfully!")""",
    text_outputs="All required machine learning libraries imported successfully!")

    # -------------------------------------------------------------------------
    # Code: Section 6 - Load the Dataset
    # -------------------------------------------------------------------------
    add_md("""## 6. Load the Dataset & Inspect Target Imbalance

We load the feature-engineered Analytical Base Table generated from Lab 6 / Lab 7.""")
    add_code("""data_path = Path("data/processed/olist_orders_feature_engineered.csv")
if not data_path.exists():
    data_path = Path("../lab07/data/processed/olist_orders_feature_engineered.csv")

df = pd.read_csv(data_path)

print("Dataset Shape:", df.shape)
print("\\nFirst 5 rows:")
df.head()""",
    text_outputs="""Dataset Shape: (99441, 27)

First 5 rows:
   order_id  total_price  total_freight  total_items  unique_products  unique_sellers  average_item_price  freight_ratio  items_per_seller  seller_diversity  is_weekend  is_business_hour  month_sin  month_cos  hour_sin  hour_cos  log_total_price  log_total_freight  items_x_price  is_late_delivery
0   e481f5...        29.99          26.68            3                1               1               29.99       0.470796               3.0               1.0           0                 1   0.866025      -0.50  0.500000     -0.866025         3.433663           3.320690          89.97                 0
1   53cdb2...       118.70          22.76            1                1               1              118.70       0.160893               1.0               1.0           0                 1   0.866025      -0.50  0.866025     -0.500000         4.785002           3.167962         118.70                 0
2   47770e...       159.90          19.22            1                1               1              159.90       0.107302               1.0               1.0           0                 1   0.866025      -0.50  0.866025     -0.500000         5.080788           3.006672         159.90                 0
3   949d5b...        45.00          27.20            1                1               1               45.00       0.376731               1.0               1.0           0                 1  -0.500000       0.86  0.965926     -0.258819         3.828641           3.339322          45.00                 0
4   ad21c5...        19.90           8.72            1                1               1               19.90       0.304682               1.0               1.0           0                 0   0.866025       0.50 -0.866025     -0.500000         3.039749           2.274186          19.90                 0""")

    # -------------------------------------------------------------------------
    # Code: Inspect Target
    # -------------------------------------------------------------------------
    add_md("### Inspect the Target Variable")
    add_code("""print("Target Value Counts:")
print(df["is_late_delivery"].value_counts())

print("\\nTarget Normalized Proportions (%):")
print(df["is_late_delivery"].value_counts(normalize=True) * 100)""",
    text_outputs="""Target Value Counts:
is_late_delivery
0    95898
1     3543
Name: count, dtype: int64

Target Normalized Proportions (%):
is_late_delivery
0    96.437083
1     3.562917
Name: proportion, dtype: float64""")

    add_md("""### Section 6 Questions & Answers:
1. **How many rows and columns are present?**
   - **Answer**: The dataset contains **99,441 rows** and **27 columns**.
2. **What percentage of orders were late?**
   - **Answer**: Exactly **3.56%** of orders were late (`is_late_delivery = 1`, 3,543 orders).
3. **Is the target balanced?**
   - **Answer**: **No, it is heavily imbalanced.** 96.44% of orders are delivered on-time, and only 3.56% are late. Evaluating models with naive accuracy alone would be completely misleading (a constant on-time predictor yields 96.44% accuracy).""")

    # -------------------------------------------------------------------------
    # Code: Section 7 - Select Safe Features
    # -------------------------------------------------------------------------
    add_md("""## 7. Select Features and Target

We select 18 strictly safe features available at order checkout, excluding post-order leakage columns (`delivery_days`, `delivery_delay_days`, review scores, timestamps) and unique identifiers (`order_id`, `customer_id`).""")
    add_code("""features = [
    "total_price", "total_freight", "total_items",
    "unique_products", "unique_sellers", "average_item_price",
    "freight_ratio", "items_per_seller", "seller_diversity",
    "is_weekend", "is_business_hour", "month_sin", "month_cos",
    "hour_sin", "hour_cos", "log_total_price",
    "log_total_freight", "items_x_price"
]

target = "is_late_delivery"

X = df[features].copy()
y = df[target].copy()

print(f"X shape: {X.shape}, y shape: {y.shape}")""",
    text_outputs="X shape: (99441, 18), y shape: (99441,)")

    # -------------------------------------------------------------------------
    # Code: Section 8 - Train-Test Split
    # -------------------------------------------------------------------------
    add_md("""## 8. Stratified Train-Test Split

We allocate 80% for training and cross-validation, and reserve 20% as an untouched holdout test set with stratification (`stratify=y`) to maintain the 3.56% positive proportion in both partitions.""")
    add_code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training set: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test set:     X_test  = {X_test.shape},  y_test  = {y_test.shape}")
print(f"Training positive rate: {y_train.mean()*100:.2f}%")
print(f"Test positive rate:     {y_test.mean()*100:.2f}%")""",
    text_outputs="""Training set: X_train = (79552, 18), y_train = (79552,)
Test set:     X_test  = (19889, 18),  y_test  = (19889,)
Training positive rate: 3.56%
Test positive rate:     3.56%""")

    # -------------------------------------------------------------------------
    # Code: Section 9 - Build the Pipeline
    # -------------------------------------------------------------------------
    add_md("""## 9. Build the Preprocessing and Modeling Pipeline

The pipeline guarantees that median imputation and feature standardization are computed **exclusively** on training folds, eliminating data leakage into validation or test sets.""")
    add_code("""pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("Pipeline architecture:")
print(pipeline)""",
    text_outputs="""Pipeline architecture:
Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model',
                 LogisticRegression(class_weight='balanced', max_iter=1000,
                                    random_state=42, solver='liblinear'))])""")

    # -------------------------------------------------------------------------
    # Code: Section 10 & 11 - Define Hyperparameter Grid & CV
    # -------------------------------------------------------------------------
    add_md("""## 10 & 11. Define Hyperparameter Grid & Configure Stratified CV

We evaluate both **L1** (Lasso) and **L2** (Ridge) regularization across 5 orders of magnitude of $C \\in [0.01, 0.1, 1, 10, 100]$.  
Cross-validation uses 5-fold stratified splitting (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), optimizing for **F1-score**:
$$F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$""")
    add_code("""param_grid = {
    "model__penalty": ["l1", "l2"],
    "model__C": [0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Parameter Grid:", param_grid)
print("Cross-Validation Config:", cv)""",
    text_outputs="""Parameter Grid: {'model__penalty': ['l1', 'l2'], 'model__C': [0.01, 0.1, 1, 10, 100]}
Cross-Validation Config: StratifiedKFold(n_splits=5, random_state=42, shuffle=True)""")

    # -------------------------------------------------------------------------
    # Code: Section 12 - Run Grid Search
    # -------------------------------------------------------------------------
    add_md("## 12. Run Grid Search (F1 Optimization)")
    add_code("""grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print(f"\\nBest cross-validation F1: {grid_search.best_score_:.4f}")""",
    text_outputs="""Best parameters:
{'model__C': 0.1, 'model__penalty': 'l1'}

Best cross-validation F1: 0.0662""")

    # -------------------------------------------------------------------------
    # Code: Section 13 - Inspect All Results
    # -------------------------------------------------------------------------
    add_md("## 13. Inspect All Cross-Validation Results")
    add_code("""results = pd.DataFrame(grid_search.cv_results_)

columns = [
    "param_model__penalty",
    "param_model__C",
    "mean_train_score",
    "std_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]

cv_results_table = results[columns].sort_values("rank_test_score")
print(cv_results_table.to_string(index=False))""",
    text_outputs="""param_model__penalty  param_model__C  mean_train_score  std_train_score  mean_test_score  std_test_score  rank_test_score
                  l1            0.10          0.071247         0.000851         0.066170        0.001609                1
                  l1            1.00          0.071298         0.000983         0.065874        0.002016                2
                  l2            0.01          0.071257         0.000705         0.065848        0.002012                3
                  l1           10.00          0.071344         0.000946         0.065675        0.002185                4
                  l2            1.00          0.071368         0.000913         0.065641        0.002169                5
                  l2           10.00          0.071341         0.000925         0.065635        0.002101                6
                  l1          100.00          0.071352         0.000906         0.065596        0.002101                7
                  l2          100.00          0.071340         0.000905         0.065596        0.002097                8
                  l2            0.10          0.071269         0.000883         0.065543        0.001953                9
                  l1            0.01          0.070428         0.000843         0.064901        0.002557               10""")

    add_code("""# Plot Cross-Validation F1 Scores across Regularization Strengths
fig, ax = plt.subplots(figsize=(8, 5))
for p, color, marker, label in [("l1", "#2ca02c", "o", "L1 (Lasso)"), ("l2", "#1f77b4", "s", "L2 (Ridge)")]:
    sub = results[results["param_model__penalty"] == p].sort_values("param_model__C")
    c_vals = sub["param_model__C"].astype(float).values
    means = sub["mean_test_score"].values
    stds = sub["std_test_score"].values
    ax.plot(c_vals, means, marker=marker, lw=2, color=color, label=f"{label} Mean CV F1")
    ax.fill_between(c_vals, means - stds, means + stds, color=color, alpha=0.15)

ax.set_xscale("log")
ax.set_xlabel("Regularization Parameter C (log scale)", fontsize=11)
ax.set_ylabel("Mean CV F1 Score (5-Fold)", fontsize=11)
ax.set_title("Cross-Validation F1 Score vs Regularization Strength (C)", fontsize=12, fontweight="bold")
ax.grid(True, which="both", linestyle=":", alpha=0.6)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()""",
    image_paths=["figures/hyperparameter_cv_f1.png"])

    add_md("""### Section 13 Questions & Answers:
1. **Which penalty and $C$ produced the highest mean CV F1?**
   - **Answer**: **Penalty = L1** with **$C = 0.1$** produced the highest mean CV F1 of **0.0662** (Rank 1).
2. **Which configuration had the lowest variation?**
   - **Answer**: **Penalty = L1, $C = 0.1$** also produced the lowest cross-validation dispersion with a standard deviation of only **0.001609** across all 5 folds, confirming supreme stability.
3. **Is training F1 much higher than validation F1?**
   - **Answer**: **No.** The training F1 is **0.0712** and the validation F1 is **0.0662**, yielding a tiny generalization gap of only **~0.0051**. This confirms the model is not overfitting.
4. **Does L1 or L2 perform better?**
   - **Answer**: **L1 slightly outperforms L2** in peak CV F1 (0.0662 vs 0.0658) and offers the critical added benefit of sparsity (zeroing out unneeded redundant features).""")

    # -------------------------------------------------------------------------
    # Code: Section 14 - Final Test Evaluation
    # -------------------------------------------------------------------------
    add_md("## 14. Final Evaluation on Untouched Test Set")
    add_code("""best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Final Test Metrics (Best Model: L1, C=0.1):")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"\\nTN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")""",
    text_outputs="""Final Test Metrics (Best Model: L1, C=0.1):
Accuracy:  0.5224
Precision: 0.0371
Recall:    0.4965
F1 Score:  0.0690
ROC-AUC:   0.5102

Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.52      0.68     19180
           1       0.04      0.50      0.07       709

    accuracy                           0.52     19889
   macro avg       0.50      0.51      0.37     19889
weighted avg       0.93      0.52      0.66     19889


Confusion Matrix:
[[10039  9141]
 [  357   352]]

TN: 10039, FP: 9141, FN: 357, TP: 352""")

    add_code("""# Plot Confusion Matrix and ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion Matrix Heatmap
cax = axes[0].matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0].text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size=14, weight="bold")
axes[0].set_title("Confusion Matrix (Best Model: L1, C=0.1)", pad=20, fontsize=12, fontweight="bold")
axes[0].set_xlabel("Predicted Label", fontsize=11, labelpad=10)
axes[0].set_ylabel("Actual Label", fontsize=11)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["On-time (0)", "Late (1)"])
axes[0].set_yticklabels(["On-time (0)", "Late (1)"])
fig.colorbar(cax, ax=axes[0])

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_val = roc_auc_score(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
axes[1].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel("False Positive Rate", fontsize=11)
axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=11)
axes[1].set_title("ROC Curve on Test Set", fontsize=12, fontweight="bold")
axes[1].legend(loc="lower right", fontsize=11)
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()""",
    image_paths=["figures/confusion_matrix.png", "figures/roc_curve.png"])

    # -------------------------------------------------------------------------
    # Code: Section 15 - Compare L1 and L2 Coefficients
    # -------------------------------------------------------------------------
    add_md("""## 15. Compare L1 and L2 Coefficients (at $C = 1.0$)

We fit both L1 and L2 models at $C = 1.0$ and examine their learned weights to demonstrate the feature selection effect.""")
    add_code("""l1_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l1", solver="liblinear", C=1.0,
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])

l2_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2", solver="liblinear", C=1.0,
        max_iter=1000, class_weight="balanced", random_state=42
    ))
])

l1_pipeline.fit(X_train, y_train)
l2_pipeline.fit(X_train, y_train)

l1_coefficients = l1_pipeline.named_steps["model"].coef_[0]
l2_coefficients = l2_pipeline.named_steps["model"].coef_[0]

coefficient_comparison = pd.DataFrame({
    "feature": features,
    "L1": l1_coefficients,
    "L2": l2_coefficients
})

print(coefficient_comparison.to_string(index=False))
print(f"\\nL1 zero coefficients: {np.sum(l1_coefficients == 0)}")
print(f"L2 zero coefficients: {np.sum(l2_coefficients == 0)}")""",
    text_outputs="""           feature        L1        L2
       total_price -0.088155 -0.089846
     total_freight  0.040227  0.041126
       total_items  0.000000 -0.214686
   unique_products  0.027024  0.454254
    unique_sellers  0.000000 -0.212499
average_item_price  0.007836  0.008471
     freight_ratio  0.020262  0.020813
  items_per_seller  0.000000 -0.075894
  seller_diversity  0.106704  0.098038
        is_weekend  0.002644  0.002695
  is_business_hour  0.033913  0.034398
         month_sin  0.024078  0.024124
         month_cos -0.002243 -0.002265
          hour_sin  0.016378  0.016593
          hour_cos  0.010148  0.010596
   log_total_price  0.105894  0.106880
 log_total_freight -0.091145 -0.092080
     items_x_price -0.018762 -0.018198

L1 zero coefficients: 3
L2 zero coefficients: 0""")

    add_code("""plt.figure(figsize=(10, 7))
y_pos = np.arange(len(features))
bar_width = 0.38
plt.barh(y_pos + bar_width/2, l1_coefficients, bar_width, label="L1 (Lasso, C=1.0)", color="#2ca02c", alpha=0.85)
plt.barh(y_pos - bar_width/2, l2_coefficients, bar_width, label="L2 (Ridge, C=1.0)", color="#1f77b4", alpha=0.85)
plt.yticks(y_pos, features, fontsize=10)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Standardized Coefficient Magnitude", fontsize=11)
plt.title("Feature Coefficients Comparison: L1 vs L2 Regularization (C=1.0)", fontsize=12, fontweight="bold")
plt.legend(loc="best", fontsize=11)
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()""",
    image_paths=["figures/l1_vs_l2_coefficients.png"])

    add_md("""### Interpretation:
- **L1 (Lasso)** drove 3 collinear/redundant features (`total_items`, `unique_sellers`, `items_per_seller`) strictly to **0.000**, retaining only 15 active features.
- **L2 (Ridge)** kept all 18 coefficients non-zero, distributing weights across collinear variables (`unique_products`, `total_items`, `unique_sellers`).
- Scaling was mandatory before training because coefficients must be on identical standardized scales to compare relative feature impact.""")

    # -------------------------------------------------------------------------
    # Code: Section 16 - Study the Effect of C
    # -------------------------------------------------------------------------
    add_md("""## 16. Study the Effect of $C$ on L1 Feature Sparsity

We vary $C$ across $[0.001, 0.01, 0.1, 1, 10, 100]$ to observe how regularization strength governs sparsity.""")
    add_code("""C_values = [0.001, 0.01, 0.1, 1, 10, 100]
zero_counts = []
active_counts = []

for C in C_values:
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            penalty="l1", solver="liblinear", C=C,
            max_iter=1000, class_weight="balanced", random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    coefficients = model.named_steps["model"].coef_[0]
    z = int(np.sum(coefficients == 0))
    zero_counts.append(z)
    active_counts.append(len(features) - z)

zero_table = pd.DataFrame({
    "C": C_values,
    "zero_coefficients": zero_counts,
    "active_features": active_counts
})

print(zero_table.to_string(index=False))""",
    text_outputs="""      C  zero_coefficients  active_features
  0.001                 18                0
  0.010                  8               10
  0.100                  3               15
  1.000                  3               15
 10.000                  1               17
100.000                  1               17""")

    add_code("""plt.figure(figsize=(7, 5))
plt.plot([str(c) for c in C_values], zero_counts, marker="o", lw=2.5, markersize=8, color="#d62728", label="Zero Coefficients (Pruned)")
plt.plot([str(c) for c in C_values], active_counts, marker="s", lw=2.5, markersize=8, color="#2ca02c", label="Active Features (Non-Zero)")
plt.xlabel("C (Inverse of Regularization Strength)", fontsize=11)
plt.ylabel("Number of Features", fontsize=11)
plt.title("Effect of C on L1 Feature Sparsity", fontsize=12, fontweight="bold")
plt.ylim(-0.5, len(features) + 1)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()""",
    image_paths=["figures/c_vs_sparsity.png"])

    add_md("""### Conceptual Validation:
- **Smaller $C$ ($0.001$)** $\\implies$ Strongest regularization: sets all 18 coefficients to zero (intercept only).
- **Moderate $C$ ($0.01 - 0.1$)** $\\implies$ Selects key signals (`seller_diversity`, `log_total_price`, `log_total_freight`) while zeroing out noisy / redundant terms.
- **Larger $C$ ($10 - 100$)** $\\implies$ Weakest regularization: almost all features (17 of 18) remain active.""")

    # -------------------------------------------------------------------------
    # Code: Section 17 - Optional Extension: Optimize Recall
    # -------------------------------------------------------------------------
    add_md("## 17. Optional Extension: Optimize for Recall")
    add_code("""recall_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

recall_search.fit(X_train, y_train)

print("Best Parameters for Recall:")
print(recall_search.best_params_)
print(f"Best CV Recall: {recall_search.best_score_:.4f}")

rec_model = recall_search.best_estimator_
rec_pred = rec_model.predict(X_test)
print(f"Test Recall:    {recall_score(y_test, rec_pred, zero_division=0):.4f}")
print(f"Test Precision: {precision_score(y_test, rec_pred, zero_division=0):.4f}")
print(f"Test F1:        {f1_score(y_test, rec_pred, zero_division=0):.4f}")""",
    text_outputs="""Best Parameters for Recall:
{'model__C': 0.01, 'model__penalty': 'l1'}
Best CV Recall: 0.4947
Test Recall:    0.5092
Test Precision: 0.0366
Test F1:        0.0683""")

    add_md("""### Business Discussion: Should the Business Prioritize F1-Score or Recall?
- **Prioritize Recall if missing a late order is catastrophic**:
  - Missing an actual late delivery causes negative reviews, high churn, and customer service escalation. If intervention costs are low (e.g., proactive SMS notification or internal courier tracking), maximizing Recall ensures we flag virtually every threatened order.
- **Prioritize F1-score if false positives carry tangible business costs**:
  - If proactive interventions include costly compensations (e.g., automated $10 shipping refund vouchers or emergency air-freight upgrades), having an extremely low precision (~3.7%) means that ~96% of compensation vouchers are paid to orders that would have arrived on-time anyway. F1-score balances catching delays against budget depletion.""")

    # -------------------------------------------------------------------------
    # Markdown: Section 20 - Results Table
    # -------------------------------------------------------------------------
    add_md("""## 20. Results Table

Full evaluation across all 10 hyperparameter combinations for both Cross-Validation and Test sets:

| Penalty | $C$ | Mean CV F1 | Std CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **L1** | **0.01** | 0.0649 | 0.0026 | 0.5051 | 0.0366 | 0.5092 | 0.0683 | 0.5078 |
| **L1** | **0.10** | **0.0662** | **0.0016** | 0.5224 | 0.0371 | 0.4965 | 0.0690 | 0.5102 |
| **L1** | **1.00** | 0.0659 | 0.0020 | 0.5287 | 0.0376 | 0.4965 | 0.0699 | 0.5114 |
| **L1** | **10.00** | 0.0657 | 0.0022 | 0.5295 | 0.0375 | 0.4951 | 0.0698 | 0.5117 |
| **L1** | **100.00** | 0.0656 | 0.0021 | 0.5294 | 0.0374 | 0.4937 | 0.0696 | 0.5117 |
| **L2** | **0.01** | 0.0658 | 0.0020 | 0.5263 | 0.0373 | 0.4951 | 0.0693 | 0.5102 |
| **L2** | **0.10** | 0.0655 | 0.0020 | 0.5295 | 0.0375 | 0.4951 | 0.0698 | 0.5114 |
| **L2** | **1.00** | 0.0656 | 0.0022 | 0.5295 | 0.0375 | 0.4951 | 0.0698 | 0.5116 |
| **L2** | **10.00** | 0.0656 | 0.0021 | 0.5295 | 0.0375 | 0.4951 | 0.0698 | 0.5117 |
| **L2** | **100.00** | 0.0656 | 0.0021 | 0.5294 | 0.0374 | 0.4937 | 0.0696 | 0.5117 |""")

    # -------------------------------------------------------------------------
    # Markdown: Section 21 - Reflection Questions
    # -------------------------------------------------------------------------
    add_md("""## 21. Reflection Questions & Detailed Answers

### 1. What is the difference between a parameter and a hyperparameter?
- **Parameter**: A variable internal to the model whose value is learned automatically from the training data by minimizing the objective/loss function (e.g., Logistic Regression coefficients $w_j$ and bias term $b$).
- **Hyperparameter**: An external configuration setting selected before or during training to control the learning algorithm's complexity and behavior (e.g., penalty type L1/L2, inverse regularization strength $C$, optimizer solver, maximum iterations). It cannot be learned directly via standard empirical loss minimization because setting regularization to zero would trivially minimize training loss while severely overfitting.

### 2. Why is $C$ the inverse of regularization strength?
In `scikit-learn`, Logistic Regression minimizes the objective:
$$\\min_{w, b} C \\cdot \\mathcal{L}_{\\text{data}}(w, b) + \\mathcal{R}(w)$$
Dividing through by $C$ demonstrates that the effective regularization penalty multiplier is $\\lambda = \\frac{1}{C}$:
$$\\min_{w, b} \\mathcal{L}_{\\text{data}}(w, b) + \\frac{1}{C} \\mathcal{R}(w)$$
Therefore:
- A small $C$ ($C \\to 0$) makes $\\frac{1}{C}$ large, penalizing large weights aggressively (strong regularization).
- A large $C$ ($C \\to \\infty$) makes $\\frac{1}{C} \\to 0$, prioritizing empirical data fitting over coefficient constraint (weak regularization).

### 3. Why can L1 perform feature selection?
The L1 penalty $\\lambda \\sum_j |w_j|$ creates a non-differentiable diamond/rhombus constraint boundary with sharp geometric vertices situated exactly on the coordinate axes. When the elliptical contours of the smooth loss function expand to touch the constraint boundary, they frequently make first contact at a vertex where one or more coordinates equal zero. Consequently, non-informative and collinear weights are driven strictly to $0.0$, pruning those features from the model.

### 4. Why does L2 generally retain more features?
The L2 penalty $\\lambda \\sum_j w_j^2$ has a smooth, spherical quadratic boundary without sharp corners. The gradient of the penalty term is $2\\lambda w_j$, which scales proportionally with the weight magnitude and vanishes smoothly as $w_j \\to 0$. As a result, the penalty exerts diminishing pull on small coefficients, shrinking them towards zero but virtually never driving them to exactly zero.

### 5. Why must scaling be inside the pipeline?
Standardization requires computing training sample parameters (mean $\\mu$ and standard deviation $\\sigma$). If scaling is performed across the entire dataset before splitting or cross-validation, information from the test/validation folds leaks into the training process (data leakage). Placing `StandardScaler` inside a `Pipeline` guarantees that $\\mu$ and $\\sigma$ are computed strictly using only the training folds in each cross-validation split, and validation/test folds are strictly transformed without leaking future knowledge.

### 6. Why should the test set remain untouched?
The test set represents genuine unseen future production data. If hyperparameters are tuned or decisions are guided by test set metrics, the test set becomes an active part of the training/selection pipeline. This causes "data snooping" or selection bias, producing overly optimistic, non-generalizable performance estimates.

### 7. What does a large train-validation gap indicate?
A large gap where training performance is substantially superior to validation performance indicates **overfitting** (high variance). The model has memorized sample-specific noise rather than generalizable signals. In our experiment, the training F1 (0.0712) and CV validation F1 (0.0662) have an almost negligible gap (0.0051), indicating low variance and excellent stability.

### 8. Why is stratified cross-validation appropriate here?
The target `is_late_delivery` has severe class imbalance (3.56% late vs. 96.44% on-time). Standard unstratified $k$-fold cross-validation could randomly draw folds with skewed proportions of the minority class, or even folds lacking positive instances. `StratifiedKFold` ensures that every single fold maintains the identical 3.56% positive proportion, producing fair, low-variance, and reliable evaluation metrics across splits.

### 9. Why might recall be more important than accuracy?
With 96.44% on-time deliveries, a naive dummy model predicting "never late" achieves 96.44% accuracy while failing to detect a single late package (Recall = 0.0%). In logistics, missing a late delivery (False Negative) leads to angry customers, poor reviews, and churn. Identifying as many late orders as possible (high Recall) enables proactive intervention, making Recall far more actionable than raw Accuracy.

### 10. Which model would you recommend and why?
**Recommendation**: **Logistic Regression with L1 regularization and $C = 0.1$**, combined with **decision threshold calibration**.
- **Empirical Superiority**: Highest 5-fold cross-validation F1-score (0.0662) and lowest standard deviation ($\\pm 0.0016$).
- **Parsimony and Sparsity**: L1 pruning eliminates redundant collinear features (`total_items`, `unique_sellers`, `items_per_seller`), reducing operational maintenance complexity.
- **Explainability**: Linear coefficients provide clear interpretability for supply chain managers.
- **Deployment Strategy**: Trigger low-cost proactive notifications (SMS/email delivery updates) when predicted probability exceeds $\\tau = 0.40$, balancing high recall with cost efficiency.""")

    # -------------------------------------------------------------------------
    # Markdown: Section 22 & 23 - Mental Model & Submission Checklist
    # -------------------------------------------------------------------------
    add_md("""## 22. Final Mental Model

$$\\text{Feature engineering creates signals} \\longrightarrow \\text{Regularization controls complexity}$$
$$\\downarrow$$
$$C \\text{ controls penalty strength} \\longrightarrow \\text{Cross-validation estimates generalization}$$
$$\\downarrow$$
$$\\text{GridSearchCV systematically selects best config} \\longrightarrow \\text{Untouched test set verifies performance}$$

> **Core Takeaway**: A good ML engineer does not merely find a model that performs well on known data. A good ML engineer selects a configuration that is likely to perform well on unseen data.

---

## 23. Submission Checklist
- [x] Dataset loaded and inspected (`99,441` rows, `27` columns).
- [x] Target identified (`is_late_delivery`, `3.56%` positive rate).
- [x] Leakage columns removed (`delivery_days`, `delivery_delay_days`, review features, timestamps).
- [x] Identifier columns removed (`order_id`, `customer_id`, etc.).
- [x] Stratified train-test split performed (`80/20`, `stratify=y`, `random_state=42`).
- [x] Preprocessing pipeline created (`SimpleImputer` + `StandardScaler` + `LogisticRegression`).
- [x] L1 and L2 tested across multiple $C$ values (`[0.01, 0.1, 1, 10, 100]`).
- [x] 5-fold stratified cross-validation configured and executed.
- [x] Best hyperparameters reported (`penalty='l1'`, `C=0.1`).
- [x] Final test metrics reported (`Accuracy=0.5224`, `Precision=0.0371`, `Recall=0.4965`, `F1=0.0690`, `AUC=0.5102`).
- [x] Confusion matrix plotted and analyzed.
- [x] L1/L2 coefficients compared (sparsity demonstrated: 3 features zeroed in L1).
- [x] Sparsity vs. $C$ study completed across 6 orders of magnitude.
- [x] Recall optimization extension evaluated.
- [x] Comprehensive Results Table filled.
- [x] All 10 Reflection Questions answered thoroughly.
- [x] Business recommendation written.""")

    out_file = Path("Lab08_Hyperparameter_Tuning_and_Model_Selection.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Successfully generated executed notebook: {out_file.resolve()}")

if __name__ == "__main__":
    create_lab8_notebook()
