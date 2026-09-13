import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 6: Feature Engineering & Feature Selection: Turning Data into ML Signals\n",
    "\n",
    "**Course**: Machine Learning  \n",
    "**Dataset**: Olist Brazilian E-Commerce Dataset  \n",
    "**Starting Point**: `data/processed/olist_orders_abt.csv`  \n",
    "**Instructor**: Sharad Laad | ORY AI Labs  \n",
    "\n",
    "---\n",
    "\n",
    "## 1. Objectives\n",
    "- Distinguish between Analytical Base Table (ABT) construction and feature engineering.\n",
    "- Engineer ratio, interaction, temporal, cyclical, binned, and log-transformed features.\n",
    "- Detect and prevent future-information data leakage.\n",
    "- Apply feature selection techniques: Variance Thresholding, Pearson Correlation filtering, Mutual Information, and Random Forest feature importance.\n",
    "- Save the final engineered dataset `data/processed/olist_orders_feature_engineered.csv`.\n",
    "- Execute a comparative mini-experiment (Baseline features vs. Engineered features) to evaluate empirical improvement."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Part A & B: Load ABT and Guard Against Data Leakage\n",
    "\n",
    "We load the ABT created in prior labs and drop:\n",
    "1. **Identifiers**: `order_id`, `customer_id`, `customer_unique_id`.\n",
    "2. **Target Leakage / Post-event columns**: `delivery_days`, `delivery_delay_days`, `review_score`, `review_comment_count`, etc."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "\n",
    "input_path = Path(\"data/processed/olist_orders_abt.csv\")\n",
    "if not input_path.exists():\n",
    "    input_path = Path(\"../lab04/data/processed/olist_orders_abt.csv\")\n",
    "\n",
    "df = pd.read_csv(input_path)\n",
    "print(\"Loaded ABT Shape:\", df.shape)\n",
    "\n",
    "target = \"is_late_delivery\"\n",
    "print(\"\\nTarget Class Distribution (%):\")\n",
    "print(df[target].value_counts(normalize=True) * 100)\n",
    "\n",
    "# Remove Identifiers and Leakage Columns\n",
    "identifier_columns = [\"order_id\", \"customer_id\", \"customer_unique_id\"]\n",
    "leakage_columns = [\n",
    "    \"delivery_days\", \"delivery_delay_days\", \"review_score\", \n",
    "    \"review_comment_count\", \"has_review_comment\", \"is_low_review\",\n",
    "    \"order_delivered_customer_date\", \"order_estimated_delivery_date\"\n",
    "]\n",
    "columns_to_drop = [c for c in identifier_columns + leakage_columns if c in df.columns]\n",
    "print(f\"\\nDropping {len(columns_to_drop)} identifiers & leakage columns.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Part C to H: Feature Engineering Implementations\n",
    "\n",
    "We now engineer:\n",
    "- **Ratios**: `average_item_price`, `freight_ratio`, `items_per_seller`, `seller_diversity`.\n",
    "- **Date/Time Indicators**: `is_weekend`, `is_business_hour`, `is_year_end`.\n",
    "- **Cyclical Encoding**: Sine and cosine components for `order_month` and `order_hour`.\n",
    "- **Log Transformations**: `log_total_price`, `log_total_freight`.\n",
    "- **Binning**: `price_band`.\n",
    "- **Interactions**: `items_x_price`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Numerical Ratios\n",
    "df[\"average_item_price\"] = df[\"total_price\"] / df[\"total_items\"].replace(0, np.nan)\n",
    "df[\"freight_ratio\"] = df[\"total_freight\"] / df[\"total_price\"].replace(0, np.nan)\n",
    "df[\"items_per_seller\"] = df[\"total_items\"] / df[\"unique_sellers\"].replace(0, np.nan)\n",
    "df[\"seller_diversity\"] = df[\"unique_sellers\"] / df[\"total_items\"].replace(0, np.nan)\n",
    "\n",
    "# Date/Time Features\n",
    "df[\"is_weekend\"] = (df[\"order_day_of_week\"] >= 5).astype(int)\n",
    "df[\"is_business_hour\"] = df[\"order_hour\"].between(9, 18).astype(int)\n",
    "df[\"is_year_end\"] = df[\"order_month\"].isin([11, 12]).astype(int)\n",
    "\n",
    "# Cyclical Encodings\n",
    "df[\"month_sin\"] = np.sin(2 * np.pi * df[\"order_month\"] / 12)\n",
    "df[\"month_cos\"] = np.cos(2 * np.pi * df[\"order_month\"] / 12)\n",
    "df[\"hour_sin\"] = np.sin(2 * np.pi * df[\"order_hour\"] / 24)\n",
    "df[\"hour_cos\"] = np.cos(2 * np.pi * df[\"order_hour\"] / 24)\n",
    "\n",
    "# Log Transformations\n",
    "df[\"log_total_price\"] = np.log1p(df[\"total_price\"].clip(lower=0))\n",
    "df[\"log_total_freight\"] = np.log1p(df[\"total_freight\"].clip(lower=0))\n",
    "\n",
    "# Binning\n",
    "df[\"price_band\"] = pd.cut(\n",
    "    df[\"total_price\"],\n",
    "    bins=[-np.inf, 100, 500, 1000, 5000, np.inf],\n",
    "    labels=[\"very_low\", \"low\", \"medium\", \"high\", \"very_high\"]\n",
    ")\n",
    "\n",
    "# Interaction\n",
    "df[\"items_x_price\"] = df[\"total_items\"] * df[\"total_price\"]\n",
    "\n",
    "print(\"Engineered Features Created Successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Part I & J: Inspect Missing Values and Summary Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "engineered_cols = [\n",
    "    \"average_item_price\", \"freight_ratio\", \"items_per_seller\", \"seller_diversity\",\n",
    "    \"is_weekend\", \"is_business_hour\", \"is_year_end\", \"month_sin\", \"month_cos\",\n",
    "    \"hour_sin\", \"hour_cos\", \"log_total_price\", \"log_total_freight\", \"items_x_price\"\n",
    "]\n",
    "print(\"Missing Values in Engineered Columns:\")\n",
    "print(df[engineered_cols].isna().sum())\n",
    "\n",
    "print(\"\\nSummary Statistics:\")\n",
    "df[engineered_cols].describe().T[[\"mean\", \"std\", \"min\", \"50%\", \"max\"]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Part K: Feature Selection Analysis\n",
    "\n",
    "We assess candidate features using:\n",
    "1. **VarianceThreshold**: Identify invariant features.\n",
    "2. **Pearson Correlation**: Detect collinear pairs ($|r| > 0.90$).\n",
    "3. **Mutual Information**: Detect non-linear dependency with target.\n",
    "4. **Random Forest Feature Importance**: Gini-based predictive signals."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn.feature_selection import VarianceThreshold, mutual_info_classif\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Prepare numerical dataset for selection\n",
    "numeric_df = df.select_dtypes(include=np.number).drop(columns=[target] + columns_to_drop, errors=\"ignore\")\n",
    "numeric_df_imputed = numeric_df.fillna(numeric_df.median())\n",
    "\n",
    "# 1. Variance Selection\n",
    "selector = VarianceThreshold(threshold=0.0)\n",
    "selector.fit(numeric_df_imputed)\n",
    "selected_var = numeric_df.columns[selector.get_support()]\n",
    "print(f\"Variance Selection: Retained {len(selected_var)} / {len(numeric_df.columns)} numeric features.\")\n",
    "\n",
    "# 2. Correlation Analysis\n",
    "corr = numeric_df_imputed.corr()\n",
    "upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\n",
    "high_corr = [(r, c, upper.loc[r, c]) for c in upper.columns for r in upper.index if pd.notna(upper.loc[r, c]) and abs(upper.loc[r, c]) > 0.90]\n",
    "print(f\"\\nHigh Correlation Pairs (|r| > 0.90):\")\n",
    "for r, c, val in high_corr:\n",
    "    print(f\"  {r} <--> {c}: r = {val:.4f}\")\n",
    "\n",
    "# 3. Mutual Information & Random Forest Importance\n",
    "sample_df = df.groupby(target, group_keys=False).apply(\n",
    "    lambda x: x.sample(min(len(x), 5000), random_state=42)\n",
    ")\n",
    "X_samp = numeric_df_imputed.loc[sample_df.index]\n",
    "y_samp = sample_df[target]\n",
    "\n",
    "mi_scores = mutual_info_classif(X_samp, y_samp, random_state=42)\n",
    "mi_df = pd.DataFrame({\"feature\": X_samp.columns, \"mi\": mi_scores}).sort_values(\"mi\", ascending=False)\n",
    "print(\"\\nTop 10 Features by Mutual Information:\")\n",
    "print(mi_df.head(10).to_string(index=False))\n",
    "\n",
    "rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight=\"balanced\", random_state=42, n_jobs=-1)\n",
    "rf.fit(X_samp, y_samp)\n",
    "rf_df = pd.DataFrame({\"feature\": X_samp.columns, \"importance\": rf.feature_importances_}).sort_values(\"importance\", ascending=False)\n",
    "print(\"\\nTop 10 Features by Random Forest Importance:\")\n",
    "print(rf_df.head(10).to_string(index=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Part P & Q: Save Engineered Feature Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "candidate_features = [\n",
    "    \"total_price\", \"total_freight\", \"total_items\", \"unique_products\", \"unique_sellers\",\n",
    "    \"order_month\", \"order_day_of_week\", \"order_hour\", \"estimated_delivery_days\",\n",
    "    \"total_payment_value\", \"max_payment_installments\", \"payment_types_count\",\n",
    "    \"average_item_price\", \"freight_ratio\", \"items_per_seller\", \"seller_diversity\",\n",
    "    \"is_weekend\", \"is_business_hour\", \"is_year_end\", \"month_sin\", \"month_cos\",\n",
    "    \"hour_sin\", \"hour_cos\", \"log_total_price\", \"log_total_freight\", \"items_x_price\"\n",
    "]\n",
    "final_features = [c for c in candidate_features if c in df.columns]\n",
    "final_df = df[final_features + [target]].copy()\n",
    "\n",
    "out_path = Path(\"data/processed/olist_orders_feature_engineered.csv\")\n",
    "final_df.to_csv(out_path, index=False)\n",
    "print(f\"Saved final engineered dataset: {out_path} with shape {final_df.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Part 25: Mini Experiment — Baseline vs. Engineered Feature Performance"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, accuracy_score, f1_score\n",
    "\n",
    "base_features = [\n",
    "    \"total_price\", \"total_freight\", \"total_items\", \"unique_products\", \"unique_sellers\",\n",
    "    \"order_month\", \"order_day_of_week\", \"order_hour\", \"estimated_delivery_days\", \"total_payment_value\"\n",
    "]\n",
    "base_features = [c for c in base_features if c in df.columns]\n",
    "\n",
    "X_base = df[base_features].fillna(df[base_features].median())\n",
    "X_eng = final_df[final_features].fillna(final_df[final_features].median())\n",
    "y = df[target]\n",
    "\n",
    "# Model A: Baseline\n",
    "X_tr_a, X_te_a, y_tr, y_te = train_test_split(X_base, y, test_size=0.20, stratify=y, random_state=42)\n",
    "clf_a = LogisticRegression(max_iter=1000, class_weight=\"balanced\", random_state=42)\n",
    "clf_a.fit(X_tr_a, y_tr)\n",
    "pred_a = clf_a.predict(X_te_a)\n",
    "\n",
    "# Model B: Engineered\n",
    "X_tr_b, X_te_b, _, _ = train_test_split(X_eng, y, test_size=0.20, stratify=y, random_state=42)\n",
    "clf_b = LogisticRegression(max_iter=1000, class_weight=\"balanced\", random_state=42)\n",
    "clf_b.fit(X_tr_b, y_tr)\n",
    "pred_b = clf_b.predict(X_te_b)\n",
    "\n",
    "print(f\"Baseline Accuracy  : {accuracy_score(y_te, pred_a):.4f} | F1-Score: {f1_score(y_te, pred_a, zero_division=0):.4f}\")\n",
    "print(f\"Engineered Accuracy: {accuracy_score(y_te, pred_b):.4f} | F1-Score: {f1_score(y_te, pred_b, zero_division=0):.4f}\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = "Lab06_Feature_Engineering_and_Selection.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Generated notebook at {output_path}")
