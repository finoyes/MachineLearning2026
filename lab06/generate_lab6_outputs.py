import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

# Ensure directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


def main():
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 70)
    log("LAB 06: FEATURE ENGINEERING & FEATURE SELECTION")
    log("=" * 70)

    # -------------------------------------------------------------------------
    # Part A: Load and Inspect the ABT
    # -------------------------------------------------------------------------
    input_path = Path("data/processed/olist_orders_abt.csv")
    if not input_path.exists():
        input_path = Path("../lab04/data/processed/olist_orders_abt.csv")

    df = pd.read_csv(input_path)
    log(f"\n[Part A] Loaded ABT from: {input_path}")
    log(f"Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

    # -------------------------------------------------------------------------
    # Part B: Separate Target and Unsafe Columns
    # -------------------------------------------------------------------------
    target = "is_late_delivery"
    log(f"\n[Part B] Target Variable: {target}")
    log("Target distribution:")
    log(df[target].value_counts().to_string())
    log(f"Proportions (%):\n{(df[target].value_counts(normalize=True) * 100).to_string()}")

    identifier_columns = [
        "order_id",
        "customer_id",
        "customer_unique_id"
    ]
    leakage_columns = [
        "delivery_days",
        "delivery_delay_days",
        "review_score",
        "review_comment_count",
        "has_review_comment",
        "is_low_review",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    columns_to_drop = [c for c in identifier_columns + leakage_columns if c in df.columns]
    log(f"\nDropping identifier & leakage columns: {columns_to_drop}")
    feature_df = df.drop(columns=columns_to_drop)
    log(f"Safe feature pool columns count: {len(feature_df.columns)}")

    # -------------------------------------------------------------------------
    # Part C: Numerical Feature Engineering
    # -------------------------------------------------------------------------
    log("\n[Part C] Computing Numerical Engineered Features...")
    if {"total_price", "total_items"}.issubset(df.columns):
        df["average_item_price"] = df["total_price"] / df["total_items"].replace(0, np.nan)
        log("- average_item_price = total_price / total_items")

    if {"total_freight", "total_price"}.issubset(df.columns):
        df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
        log("- freight_ratio = total_freight / total_price")

    if {"total_items", "unique_sellers"}.issubset(df.columns):
        df["items_per_seller"] = df["total_items"] / df["unique_sellers"].replace(0, np.nan)
        log("- items_per_seller = total_items / unique_sellers")

    if {"unique_sellers", "total_items"}.issubset(df.columns):
        df["seller_diversity"] = df["unique_sellers"] / df["total_items"].replace(0, np.nan)
        log("- seller_diversity = unique_sellers / total_items")

    # -------------------------------------------------------------------------
    # Part D: Date and Time Feature Engineering
    # -------------------------------------------------------------------------
    log("\n[Part D] Computing Date and Time Features...")
    if "order_day_of_week" in df.columns:
        df["is_weekend"] = (df["order_day_of_week"] >= 5).astype(int)
        log("- is_weekend (order_day_of_week >= 5)")

    if "order_hour" in df.columns:
        df["is_business_hour"] = df["order_hour"].between(9, 18).astype(int)
        log("- is_business_hour (order_hour between 9 and 18)")

    if "order_month" in df.columns:
        df["is_year_end"] = df["order_month"].isin([11, 12]).astype(int)
        log("- is_year_end (order_month in [11, 12])")

    # -------------------------------------------------------------------------
    # Part E: Cyclical Features
    # -------------------------------------------------------------------------
    log("\n[Part E] Computing Cyclical Features (Sine/Cosine)...")
    if "order_month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["order_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["order_month"] / 12)
        log("- month_sin, month_cos")

    if "order_hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
        log("- hour_sin, hour_cos")

    # -------------------------------------------------------------------------
    # Part F: Log Transformation
    # -------------------------------------------------------------------------
    log("\n[Part F] Applying Log Transformations (log1p)...")
    if "total_price" in df.columns:
        df["log_total_price"] = np.log1p(df["total_price"].clip(lower=0))
        log("- log_total_price = log1p(total_price)")

    if "total_freight" in df.columns:
        df["log_total_freight"] = np.log1p(df["total_freight"].clip(lower=0))
        log("- log_total_freight = log1p(total_freight)")

    # -------------------------------------------------------------------------
    # Part G: Binning
    # -------------------------------------------------------------------------
    log("\n[Part G] Applying Binning to total_price...")
    if "total_price" in df.columns:
        df["price_band"] = pd.cut(
            df["total_price"],
            bins=[-np.inf, 100, 500, 1000, 5000, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"]
        )
        log("- price_band distribution:\n" + df["price_band"].value_counts().to_string())

    # -------------------------------------------------------------------------
    # Part H: Interaction Features
    # -------------------------------------------------------------------------
    log("\n[Part H] Computing Interaction Features...")
    if {"total_items", "total_price"}.issubset(df.columns):
        df["items_x_price"] = df["total_items"] * df["total_price"]
        log("- items_x_price = total_items * total_price")

    # -------------------------------------------------------------------------
    # Part I & J: Inspect Invalid Values and Engineered Features
    # -------------------------------------------------------------------------
    engineered_columns = [
        "average_item_price",
        "freight_ratio",
        "items_per_seller",
        "seller_diversity",
        "is_weekend",
        "is_business_hour",
        "is_year_end",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "log_total_price",
        "log_total_freight",
        "price_band",
        "items_x_price"
    ]
    available_engineered = [c for c in engineered_columns if c in df.columns]
    log("\n[Part I] Missing Values in Engineered Features:")
    log(df[available_engineered].isna().sum().to_string())

    num_engineered = [c for c in available_engineered if c != "price_band"]
    log("\n[Part J] Summary Statistics of Numerical Engineered Features:")
    log(df[num_engineered].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    # -------------------------------------------------------------------------
    # Part K: Feature Selection Techniques
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("[Part K] FEATURE SELECTION")
    log("=" * 50)

    # 17.1 Variance-Based Selection
    numeric_df = df.select_dtypes(include=np.number).copy()
    numeric_df = numeric_df.drop(columns=[target] + columns_to_drop, errors="ignore")
    # Impute missing values with median for selection algorithms
    numeric_df_imputed = numeric_df.fillna(numeric_df.median())

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(numeric_df_imputed)
    selected_variance = numeric_df.columns[selector.get_support()].tolist()
    log(f"17.1 Variance Filtering (threshold=0.0):")
    log(f"   Original Numeric Count: {len(numeric_df.columns)}")
    log(f"   Retained Numeric Count: {len(selected_variance)}")
    zero_variance_cols = set(numeric_df.columns) - set(selected_variance)
    log(f"   Dropped Zero-Variance Columns: {list(zero_variance_cols)}")

    # 18 Part L: Correlation-Based Selection
    log("\n18. Correlation-Based Selection (Threshold = 0.90):")
    corr = numeric_df_imputed.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        for row in upper.index:
            val = upper.loc[row, col]
            if pd.notna(val) and abs(val) > 0.90:
                high_corr_pairs.append((row, col, float(val)))

    log(f"Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.90):")
    for r, c, v in high_corr_pairs:
        log(f"   - {r:<22} <---> {c:<22} : r = {v:.4f}")

    # Plot Correlation Matrix Heatmap for top features
    plt.figure(figsize=(12, 10))
    sample_corr_cols = [
        "total_price", "log_total_price", "total_freight", "log_total_freight",
        "total_items", "unique_sellers", "items_per_seller", "seller_diversity",
        "average_item_price", "items_x_price", "is_weekend", "month_sin", "hour_sin"
    ]
    sample_corr_cols = [c for c in sample_corr_cols if c in numeric_df_imputed.columns]
    sub_corr = numeric_df_imputed[sample_corr_cols].corr()
    im = plt.imshow(sub_corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(sample_corr_cols)), sample_corr_cols, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(sample_corr_cols)), sample_corr_cols, fontsize=9)
    plt.title("Correlation Heatmap of Key Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/correlation_heatmap.png", dpi=300)
    plt.close()
    log("Saved correlation heatmap to figures/correlation_heatmap.png")

    # 19 Part M: Mutual Information
    log("\n19. Mutual Information (Target: is_late_delivery):")
    # Sample 15,000 for responsive execution while maintaining high statistical power
    sample_size = min(15000, len(df))
    sample_idx = df.groupby(target, group_keys=False).apply(
        lambda x: x.sample(int(sample_size * len(x) / len(df)), random_state=42)
    ).index
    mi_sample_X = numeric_df_imputed.loc[sample_idx]
    mi_sample_y = df.loc[sample_idx, target]

    mi_scores = mutual_info_classif(mi_sample_X, mi_sample_y, random_state=42)
    mi_results = pd.DataFrame({
        "feature": mi_sample_X.columns,
        "mutual_information": mi_scores
    }).sort_values("mutual_information", ascending=False)

    log("Top 15 Features by Mutual Information:")
    log(mi_results.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    top_mi = mi_results.head(12)
    plt.barh(top_mi["feature"][::-1], top_mi["mutual_information"][::-1], color="#3b82f6")
    plt.xlabel("Mutual Information Score", fontsize=11)
    plt.title("Top 12 Features by Mutual Information with Target", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/mutual_information.png", dpi=300)
    plt.close()
    log("Saved mutual information plot to figures/mutual_information.png")

    # 20 Part N: Model-Based Feature Importance (Random Forest)
    log("\n20. Random Forest Feature Importance:")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=12
    )
    rf.fit(mi_sample_X, mi_sample_y)

    rf_importance = pd.DataFrame({
        "feature": mi_sample_X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    log("Top 15 Features by Random Forest Importance:")
    log(rf_importance.head(15).to_string(index=False))

    plt.figure(figsize=(10, 6))
    top_rf = rf_importance.head(12)
    plt.barh(top_rf["feature"][::-1], top_rf["importance"][::-1], color="#10b981")
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.title("Top 12 Features by Random Forest Importance", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=300)
    plt.close()
    log("Saved feature importance plot to figures/feature_importance.png")

    # -------------------------------------------------------------------------
    # Part P & Q: Final Feature Set and Saving Dataset
    # -------------------------------------------------------------------------
    candidate_features = [
        "total_price",
        "total_freight",
        "total_items",
        "unique_products",
        "unique_sellers",
        "order_month",
        "order_day_of_week",
        "order_hour",
        "estimated_delivery_days",
        "total_payment_value",
        "max_payment_installments",
        "payment_types_count",
        "average_item_price",
        "freight_ratio",
        "items_per_seller",
        "seller_diversity",
        "is_weekend",
        "is_business_hour",
        "is_year_end",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "log_total_price",
        "log_total_freight",
        "items_x_price"
    ]
    final_features = [c for c in candidate_features if c in df.columns]
    final_df = df[final_features + [target]].copy()

    log(f"\n[Part P] Final Feature Count: {len(final_features)}")
    log(f"Final Dataset Shape: {final_df.shape}")

    output_path = Path("data/processed/olist_orders_feature_engineered.csv")
    final_df.to_csv(output_path, index=False)
    log(f"[Part Q] Saved engineered dataset to: {output_path}")

    # -------------------------------------------------------------------------
    # Part 25: Mini Experiment — Did Feature Engineering Add Value?
    # -------------------------------------------------------------------------
    log("\n" + "=" * 50)
    log("Part 25: MINI EXPERIMENT (Baseline vs. Engineered)")
    log("=" * 50)

    # Base features
    base_cols = [
        "total_price", "total_freight", "total_items",
        "unique_products", "unique_sellers", "order_month",
        "order_day_of_week", "order_hour", "estimated_delivery_days",
        "total_payment_value"
    ]
    base_cols = [c for c in base_cols if c in df.columns]

    # Engineered feature set includes base + engineered
    eng_cols = final_features

    # Prepare datasets
    X_base = df[base_cols].fillna(df[base_cols].median())
    X_eng = df[eng_cols].fillna(df[eng_cols].median())
    y = df[target]

    # Model A: Baseline
    X_tr_a, X_te_a, y_tr, y_te = train_test_split(
        X_base, y, test_size=0.20, random_state=42, stratify=y
    )
    clf_a = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_a.fit(X_tr_a, y_tr)
    y_pred_a = clf_a.predict(X_te_a)
    acc_a = accuracy_score(y_te, y_pred_a)
    f1_a = f1_score(y_te, y_pred_a, zero_division=0)

    # Model B: Engineered
    X_tr_b, X_te_b, _, _ = train_test_split(
        X_eng, y, test_size=0.20, random_state=42, stratify=y
    )
    clf_b = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf_b.fit(X_tr_b, y_tr)
    y_pred_b = clf_b.predict(X_te_b)
    acc_b = accuracy_score(y_te, y_pred_b)
    f1_b = f1_score(y_te, y_pred_b, zero_division=0)

    log(f"{'Experiment':<25} | {'Accuracy':<12} | {'F1-Score':<12}")
    log("-" * 55)
    log(f"{'Model A (Baseline)':<25} | {acc_a:<12.4f} | {f1_a:<12.4f}")
    log(f"{'Model B (Engineered)':<25} | {acc_b:<12.4f} | {f1_b:<12.4f}")

    log("\nBaseline Classification Report:")
    log(classification_report(y_te, y_pred_a, zero_division=0))

    log("Engineered Classification Report:")
    log(classification_report(y_te, y_pred_b, zero_division=0))

    with open("output_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log("\nSaved complete output log to output_log.txt")


if __name__ == "__main__":
    main()
