import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("LAB 07: MODEL EVALUATION & CROSS-VALIDATION")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # Part A: Load the Data
    # -------------------------------------------------------------------------
    input_path = Path("data/processed/olist_orders_feature_engineered.csv")
    if not input_path.exists():
        input_path = Path("../lab06/data/processed/olist_orders_feature_engineered.csv")

    df = pd.read_csv(input_path)
    log(f"\n[Part A] Loaded dataset from: {input_path}")
    log(f"Shape: {df.shape}")

    target = "is_late_delivery"
    log(f"Target column: {target}")
    log("Target distribution counts:")
    log(df[target].value_counts().to_string())
    log("\nTarget distribution proportions (%):")
    log((df[target].value_counts(normalize=True) * 100).to_string())

    # -------------------------------------------------------------------------
    # Part B: Define X and y
    # -------------------------------------------------------------------------
    y = df[target]
    X = df.drop(columns=[target]).select_dtypes(include=np.number)
    # Handle missing values for simple experiment
    X = X.fillna(X.median())

    log(f"\n[Part B] Features matrix X shape: {X.shape}")
    log(f"Target vector y shape: {y.shape}")

    # -------------------------------------------------------------------------
    # Part C: Train-Test Split
    # -------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    log(f"\n[Part C] Train-Test Split (80/20, stratify=y, random_state=42):")
    log(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    log(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

    # -------------------------------------------------------------------------
    # Part D: Train Baseline Model
    # -------------------------------------------------------------------------
    baseline_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred = baseline_model.predict(X_test)
    y_prob = baseline_model.predict_proba(X_test)[:, 1]

    log("\n[Part D] Trained Baseline Logistic Regression (class_weight='balanced')")

    # -------------------------------------------------------------------------
    # Part E: Confusion Matrix
    # -------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    log("\n[Part E] Confusion Matrix:")
    log(f"{'':<15} | {'Predicted 0 (On-time)':<22} | {'Predicted 1 (Late)':<20}")
    log("-" * 65)
    log(f"{'Actual 0':<15} | {tn:<22} (TN) | {fp:<20} (FP)")
    log(f"{'Actual 1':<15} | {fn:<22} (FN) | {tp:<20} (TP)")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix (Baseline Logistic Regression)", fontsize=11, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["On-time (0)", "Late (1)"])
    plt.yticks(tick_marks, ["On-time (0)", "Late (1)"])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            plt.text(j, i, f"{val}\n({val/cm.sum()*100:.1f}%)", ha="center", va="center", color=color, fontsize=11)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=300)
    plt.close()
    log("Saved confusion matrix plot to figures/confusion_matrix.png")

    # -------------------------------------------------------------------------
    # Part F - J: Metrics and Classification Report
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    log("\n[Part F - K] Evaluation Metrics (Test Set):")
    log(f"Accuracy  : {acc:.4f}")
    log(f"Precision : {prec:.4f}")
    log(f"Recall    : {rec:.4f}")
    log(f"F1-Score  : {f1:.4f}")
    log(f"ROC-AUC   : {roc_auc:.4f}")

    log("\nClassification Report:")
    log(classification_report(y_test, y_pred, zero_division=0))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/roc_curve.png", dpi=300)
    plt.close()
    log("Saved ROC curve to figures/roc_curve.png")

    # -------------------------------------------------------------------------
    # Section 17 & Challenge Exercise (Section 29): Classification Thresholds
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Section 17 & 29] CLASSIFICATION THRESHOLD ANALYSIS")
    log("=" * 50)
    thresholds = [0.30, 0.40, 0.50, 0.70]
    thresh_table = []
    for th in thresholds:
        th_pred = (y_prob >= th).astype(int)
        p_th = precision_score(y_test, th_pred, zero_division=0)
        r_th = recall_score(y_test, th_pred, zero_division=0)
        f_th = f1_score(y_test, th_pred, zero_division=0)
        thresh_table.append({"Threshold": th, "Precision": p_th, "Recall": r_th, "F1": f_th})

    thresh_df = pd.DataFrame(thresh_table)
    log(thresh_df.to_string(index=False))

    # Plot Threshold Trade-off
    all_th = np.linspace(0.1, 0.9, 50)
    th_precs = [precision_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in all_th]
    th_recs = [recall_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in all_th]
    th_f1s = [f1_score(y_test, (y_prob >= t).astype(int), zero_division=0) for t in all_th]

    plt.figure(figsize=(8, 5))
    plt.plot(all_th, th_precs, label="Precision", color="#059669", lw=2)
    plt.plot(all_th, th_recs, label="Recall", color="#dc2626", lw=2)
    plt.plot(all_th, th_f1s, label="F1-Score", color="#4f46e5", lw=2, linestyle="--")
    plt.axvline(0.50, color="gray", linestyle=":", label="Default Threshold (0.50)")
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("Precision-Recall-F1 Trade-off Across Classification Thresholds", fontsize=12, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/threshold_tradeoff.png", dpi=300)
    plt.close()
    log("Saved threshold trade-off plot to figures/threshold_tradeoff.png")

    # -------------------------------------------------------------------------
    # Part M & N: 5-Fold Stratified Cross-Validation
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Part M & N] 5-FOLD STRATIFIED CROSS-VALIDATION")
    log("=" * 50)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring="f1")

    log("Fold F1 Scores:")
    for idx, sc in enumerate(cv_scores, 1):
        log(f"Fold {idx}: {sc:.4f}")
    log(f"Mean F1  : {cv_scores.mean():.4f}")
    log(f"Std F1   : {cv_scores.std():.4f}")

    # -------------------------------------------------------------------------
    # Part O: Compare Multiple Metrics with Cross-Validation
    # -------------------------------------------------------------------------
    log("\n[Part O] Multi-Metric Cross-Validation Summary:")
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_multi = cross_validate(baseline_model, X_train, y_train, cv=cv, scoring=scoring)
    for m in scoring:
        vals = cv_multi["test_" + m]
        log(f"{m:<12} | Mean = {vals.mean():.4f} | Std = {vals.std():.4f}")

    # Plot CV Scores across folds
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, 6), cv_scores, marker="o", color="#8b5cf6", lw=2, markersize=8)
    plt.axhline(cv_scores.mean(), color="#ef4444", linestyle="--", label=f"Mean F1 ({cv_scores.mean():.4f})")
    plt.fill_between(range(1, 6), cv_scores.mean() - cv_scores.std(), cv_scores.mean() + cv_scores.std(), color="#8b5cf6", alpha=0.15, label="±1 Std Dev")
    plt.title("5-Fold Stratified Cross-Validation F1 Scores", fontsize=12, fontweight="bold")
    plt.xlabel("Fold Number", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/cv_scores.png", dpi=300)
    plt.close()
    log("Saved CV scores plot to figures/cv_scores.png")

    # -------------------------------------------------------------------------
    # Part P: Compare Baseline and Engineered Features via CV
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Part P] BASELINE VS. ENGINEERED FEATURES UNDER CV")
    log("=" * 50)
    base_cols = [
        "total_price", "total_freight", "total_items",
        "unique_products", "unique_sellers", "order_month",
        "order_day_of_week", "order_hour", "estimated_delivery_days",
        "total_payment_value"
    ]
    base_cols = [c for c in base_cols if c in X.columns]

    X_train_base = X_train[base_cols]
    cv_scores_base = cross_val_score(baseline_model, X_train_base, y_train, cv=cv, scoring="f1")
    cv_scores_eng = cv_scores

    log(f"{'Experiment':<25} | {'Mean F1':<12} | {'Std F1':<12}")
    log("-" * 55)
    log(f"{'Model A (Baseline)':<25} | {cv_scores_base.mean():<12.4f} | {cv_scores_base.std():<12.4f}")
    log(f"{'Model B (Engineered)':<25} | {cv_scores_eng.mean():<12.4f} | {cv_scores_eng.std():<12.4f}")

    # -------------------------------------------------------------------------
    # Part Q & Section 26: Pipeline-Based Evaluation & Unseen Test Set
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Part Q & Section 26] PIPELINE EVALUATION (Zero Data Leakage)")
    log("=" * 50)
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])

    pipe_cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    log(f"Pipeline CV F1 Scores: Mean = {pipe_cv_scores.mean():.4f}, Std = {pipe_cv_scores.std():.4f}")

    # Final fit on training data and one-shot evaluation on untouched test set
    pipeline.fit(X_train, y_train)
    final_pred = pipeline.predict(X_test)
    final_prob = pipeline.predict_proba(X_test)[:, 1]

    final_acc = accuracy_score(y_test, final_pred)
    final_prec = precision_score(y_test, final_pred, zero_division=0)
    final_rec = recall_score(y_test, final_pred, zero_division=0)
    final_f1 = f1_score(y_test, final_pred, zero_division=0)
    final_auc = roc_auc_score(y_test, final_prob)

    log("\nFinal Pipeline Evaluation on Untouched Test Set:")
    log(f"Accuracy  : {final_acc:.4f}")
    log(f"Precision : {final_prec:.4f}")
    log(f"Recall    : {final_rec:.4f}")
    log(f"F1-Score  : {final_f1:.4f}")
    log(f"ROC-AUC   : {final_auc:.4f}")

    log("\nFinal Classification Report:")
    log(classification_report(y_test, final_pred, zero_division=0))

    # -------------------------------------------------------------------------
    # Section 27: Recommended Evaluation Report Summary
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Section 27] RECOMMENDED EVALUATION REPORT SUMMARY")
    log("=" * 50)
    eval_summary = pd.DataFrame([
        {"Metric": "Accuracy", "Result": f"{final_acc:.4f}"},
        {"Metric": "Precision", "Result": f"{final_prec:.4f}"},
        {"Metric": "Recall", "Result": f"{final_rec:.4f}"},
        {"Metric": "F1-score", "Result": f"{final_f1:.4f}"},
        {"Metric": "ROC-AUC", "Result": f"{final_auc:.4f}"},
        {"Metric": "CV Mean F1", "Result": f"{pipe_cv_scores.mean():.4f}"},
        {"Metric": "CV Std F1", "Result": f"{pipe_cv_scores.std():.4f}"}
    ])
    log(eval_summary.to_string(index=False))

    with open("output_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("\nSaved complete output log to output_log.txt")


if __name__ == "__main__":
    main()
