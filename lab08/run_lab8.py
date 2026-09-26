import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Ensure output directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 80)
    log("LAB 08: HYPERPARAMETER TUNING AND MODEL SELECTION USING CROSS-VALIDATION")
    log("=" * 80)

    # -------------------------------------------------------------------------
    # Section 6: Load the Dataset
    # -------------------------------------------------------------------------
    data_path = Path("data/processed/olist_orders_feature_engineered.csv")
    if not data_path.exists():
        data_path = Path("../lab07/data/processed/olist_orders_feature_engineered.csv")
    
    df = pd.read_csv(data_path)
    log(f"\n[Section 6] Loaded dataset from: {data_path}")
    log(f"Dataset shape: {df.shape}")
    log(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    
    target = "is_late_delivery"
    target_counts = df[target].value_counts()
    target_props = df[target].value_counts(normalize=True) * 100

    log("\nTarget Counts (is_late_delivery):")
    log(target_counts.to_string())
    log("\nTarget Proportions (%):")
    log(target_props.to_string())

    pct_late = target_props[1]
    log(f"\nQuestion 1 Answer: {df.shape[0]} rows and {df.shape[1]} columns are present.")
    log(f"Question 2 Answer: {pct_late:.2f}% of orders were delivered late.")
    log(f"Question 3 Answer: The target is heavily imbalanced ({pct_late:.2f}% minority class).")

    # -------------------------------------------------------------------------
    # Section 7: Select Features and Target
    # -------------------------------------------------------------------------
    features = [
        "total_price", "total_freight", "total_items",
        "unique_products", "unique_sellers", "average_item_price",
        "freight_ratio", "items_per_seller", "seller_diversity",
        "is_weekend", "is_business_hour", "month_sin", "month_cos",
        "hour_sin", "hour_cos", "log_total_price",
        "log_total_freight", "items_x_price"
    ]

    log(f"\n[Section 7] Selected {len(features)} safe features:")
    for f in features:
        log(f"  - {f}")
    
    X = df[features].copy()
    y = df[target].copy()

    # -------------------------------------------------------------------------
    # Section 8: Train-Test Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    log(f"\n[Section 8] Train-Test Split (80/20, stratify=y, random_state=42):")
    log(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    log(f"  X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    log(f"  Train Late Proportion: {y_train.mean()*100:.2f}%")
    log(f"  Test Late Proportion:  {y_test.mean()*100:.2f}%")

    # -------------------------------------------------------------------------
    # Section 9: Build the Pipeline
    # -------------------------------------------------------------------------
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    log("\n[Section 9] Pipeline successfully constructed:")
    log("  Step 1: SimpleImputer(strategy='median')")
    log("  Step 2: StandardScaler()")
    log("  Step 3: LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')")

    # -------------------------------------------------------------------------
    # Section 10 & 11: Define Hyperparameter Grid & Cross-Validation
    # -------------------------------------------------------------------------
    param_grid = {
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1, 10, 100]
    }
    log(f"\n[Section 10] Hyperparameter Grid: {param_grid}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    log(f"\n[Section 11] Cross-Validation Config: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)")

    # -------------------------------------------------------------------------
    # Section 12: Run Grid Search (F1 Optimization)
    # -------------------------------------------------------------------------
    log("\n[Section 12] Running GridSearchCV optimizing for F1-score...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    log("\nBest parameters:")
    log(str(best_params))
    log(f"\nBest cross-validation F1: {best_score:.4f}")

    # -------------------------------------------------------------------------
    # Section 13: Inspect All Results
    # -------------------------------------------------------------------------
    results = pd.DataFrame(grid_search.cv_results_)
    columns = [
        "param_model__penalty",
        "param_model__C",
        "mean_train_score",
        "std_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
    sorted_results = results[columns].sort_values("rank_test_score")
    log("\n[Section 13] Cross-Validation Results Table:")
    log(sorted_results.to_string(index=False))

    # Questions from Section 13
    best_row = sorted_results.iloc[0]
    min_var_row = sorted_results.sort_values("std_test_score").iloc[0]
    log(f"\nQuestion 1 Answer: Penalty = {best_row['param_model__penalty']} and C = {best_row['param_model__C']} produced the highest mean CV F1 ({best_row['mean_test_score']:.4f}).")
    log(f"Question 2 Answer: Configuration penalty = {min_var_row['param_model__penalty']} and C = {min_var_row['param_model__C']} had the lowest variation (std = {min_var_row['std_test_score']:.6f}).")
    log(f"Question 3 Answer: Training F1 ({best_row['mean_train_score']:.4f}) is closely matched with validation F1 ({best_row['mean_test_score']:.4f}), difference is ~{abs(best_row['mean_train_score'] - best_row['mean_test_score']):.4f}, indicating minimal overfitting.")
    l1_best = results[results["param_model__penalty"] == "l1"]["mean_test_score"].max()
    l2_best = results[results["param_model__penalty"] == "l2"]["mean_test_score"].max()
    log(f"Question 4 Answer: L1 best mean CV F1 = {l1_best:.4f}, L2 best mean CV F1 = {l2_best:.4f}. L1 and L2 achieve very similar performance with {'L1' if l1_best >= l2_best else 'L2'} slightly leading.")

    # -------------------------------------------------------------------------
    # Section 14: Final Test Evaluation
    # -------------------------------------------------------------------------
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, y_prob)

    log("\n[Section 14] Final Test Evaluation of Best Model:")
    log(f"Accuracy:  {test_acc:.4f}")
    log(f"Precision: {test_prec:.4f}")
    log(f"Recall:    {test_rec:.4f}")
    log(f"F1:        {test_f1:.4f}")
    log(f"ROC-AUC:   {test_auc:.4f}")

    log("\nClassification Report:")
    log(classification_report(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log("\nConfusion Matrix:")
    log(str(cm))
    log(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=f"{cm[i, j]:,}", va="center", ha="center", size=14, weight="bold")
    plt.title("Confusion Matrix (Best Model on Test Set)", pad=20, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("Actual Label", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["On-time (0)", "Late (1)"])
    ax.set_yticklabels(["On-time (0)", "Late (1)"])
    fig.colorbar(cax)
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png")
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC curve (AUC = {test_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/roc_curve.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Section 15: Compare L1 and L2 Coefficients (at C=1.0)
    # -------------------------------------------------------------------------
    l1_pipeline = Pipeline(steps=[
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

    coef_comp = pd.DataFrame({
        "feature": features,
        "L1": l1_coefficients,
        "L2": l2_coefficients
    })

    log("\n[Section 15] Coefficient Comparison (C=1.0):")
    log(coef_comp.to_string(index=False))
    l1_zeros = int(np.sum(l1_coefficients == 0))
    l2_zeros = int(np.sum(l2_coefficients == 0))
    log(f"L1 zero coefficients: {l1_zeros}")
    log(f"L2 zero coefficients: {l2_zeros}")

    # Plot Coefficient Comparison
    plt.figure(figsize=(10, 7), dpi=300)
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
    plt.savefig("figures/l1_vs_l2_coefficients.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Section 16: Study the Effect of C on Sparsity
    # -------------------------------------------------------------------------
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    zero_counts = []
    non_zero_counts = []

    for c_val in C_values:
        m = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l1", solver="liblinear", C=c_val,
                max_iter=1000, class_weight="balanced", random_state=42
            ))
        ])
        m.fit(X_train, y_train)
        coefs = m.named_steps["model"].coef_[0]
        z = int(np.sum(coefs == 0))
        zero_counts.append(z)
        non_zero_counts.append(len(features) - z)

    zero_table = pd.DataFrame({
        "C": C_values,
        "zero_coefficients": zero_counts,
        "active_features": non_zero_counts
    })
    log("\n[Section 16] Study Effect of C on L1 Sparsity:")
    log(zero_table.to_string(index=False))

    # Plot C vs Sparsity
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot([str(c) for c in C_values], zero_counts, marker="o", lw=2.5, markersize=8, color="#d62728", label="Zero Coefficients")
    plt.plot([str(c) for c in C_values], non_zero_counts, marker="s", lw=2.5, markersize=8, color="#2ca02c", label="Active Features (Non-Zero)")
    plt.xlabel("C (Inverse of Regularization Strength)", fontsize=11)
    plt.ylabel("Number of Features", fontsize=11)
    plt.title("Effect of C on L1 Feature Sparsity", fontsize=12, fontweight="bold")
    plt.ylim(-0.5, len(features) + 1)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/c_vs_sparsity.png")
    plt.close()

    # -------------------------------------------------------------------------
    # Section 17: Optional Extension: Optimize Recall
    # -------------------------------------------------------------------------
    log("\n[Section 17] Running Optional Extension: GridSearchCV optimizing Recall...")
    recall_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="recall",
        n_jobs=-1,
        return_train_score=True
    )
    recall_search.fit(X_train, y_train)
    log(f"Recall Search Best Parameters: {recall_search.best_params_}")
    log(f"Recall Search Best CV Recall: {recall_search.best_score_:.4f}")

    rec_model = recall_search.best_estimator_
    rec_pred = rec_model.predict(X_test)
    log(f"Recall-Optimized Model Test Metrics: Accuracy={accuracy_score(y_test, rec_pred):.4f}, "
        f"Precision={precision_score(y_test, rec_pred, zero_division=0):.4f}, "
        f"Recall={recall_score(y_test, rec_pred, zero_division=0):.4f}, "
        f"F1={f1_score(y_test, rec_pred, zero_division=0):.4f}")

    # -------------------------------------------------------------------------
    # Section 20: Comprehensive Results Table
    # -------------------------------------------------------------------------
    # We evaluate all 10 configurations across penalty in ['l1', 'l2'] and C in [0.01, 0.1, 1, 10, 100]
    # For CV, we extract Mean CV F1 and Std CV F1 from grid_search.cv_results_
    # For Test metrics, we evaluate each fitted configuration pipeline on X_test, y_test
    log("\n[Section 20] Building Full Results Table for Deliverables...")

    table_rows = []
    # Plot CV curve as well
    cv_plot_data = {"l1": {"C": [], "mean": [], "std": []}, "l2": {"C": [], "mean": [], "std": []}}

    for p in ["l1", "l2"]:
        for c_val in [0.01, 0.1, 1, 10, 100]:
            # find corresponding row in grid_search.cv_results_
            match_mask = (results["param_model__penalty"] == p) & (results["param_model__C"] == c_val)
            cv_row = results[match_mask].iloc[0]
            mean_cv_f1 = cv_row["mean_test_score"]
            std_cv_f1 = cv_row["std_test_score"]

            cv_plot_data[p]["C"].append(c_val)
            cv_plot_data[p]["mean"].append(mean_cv_f1)
            cv_plot_data[p]["std"].append(std_cv_f1)

            # Fit single pipeline on X_train and evaluate on X_test
            pipe_eval = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    penalty=p, solver="liblinear", C=c_val,
                    max_iter=1000, class_weight="balanced", random_state=42
                ))
            ])
            pipe_eval.fit(X_train, y_train)
            test_preds = pipe_eval.predict(X_test)
            test_probs = pipe_eval.predict_proba(X_test)[:, 1]

            t_acc = accuracy_score(y_test, test_preds)
            t_prec = precision_score(y_test, test_preds, zero_division=0)
            t_rec = recall_score(y_test, test_preds, zero_division=0)
            t_f1 = f1_score(y_test, test_preds, zero_division=0)
            t_auc = roc_auc_score(y_test, test_probs)

            table_rows.append({
                "Penalty": p.upper(),
                "C": c_val,
                "Mean CV F1": round(mean_cv_f1, 4),
                "Std CV F1": round(std_cv_f1, 4),
                "Test Accuracy": round(t_acc, 4),
                "Test Precision": round(t_prec, 4),
                "Test Recall": round(t_rec, 4),
                "Test F1": round(t_f1, 4),
                "Test ROC-AUC": round(t_auc, 4)
            })

    results_table_df = pd.DataFrame(table_rows)
    log("\nFinal Section 20 Results Table:")
    log(results_table_df.to_string(index=False))

    # Plot Hyperparameter CV F1 Curve
    plt.figure(figsize=(8, 5), dpi=300)
    for p, color, marker, label in [("l1", "#2ca02c", "o", "L1 (Lasso)"), ("l2", "#1f77b4", "s", "L2 (Ridge)")]:
        means = np.array(cv_plot_data[p]["mean"])
        stds = np.array(cv_plot_data[p]["std"])
        c_vals = np.array(cv_plot_data[p]["C"])
        plt.plot(c_vals, means, marker=marker, lw=2, color=color, label=f"{label} Mean CV F1")
        plt.fill_between(c_vals, means - stds, means + stds, color=color, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("Regularization Parameter C (log scale)", fontsize=11)
    plt.ylabel("Mean CV F1 Score (5-Fold)", fontsize=11)
    plt.title("Cross-Validation F1 Score vs Regularization Strength (C)", fontsize=12, fontweight="bold")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/hyperparameter_cv_f1.png")
    plt.close()

    # Save output log
    with open("output_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    log("\nAll experiments, outputs, and figures generated successfully!")

if __name__ == "__main__":
    main()
