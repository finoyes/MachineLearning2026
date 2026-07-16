"""
Lab 2: Build Analytical Base Table — Generate all outputs
Produces:
  data/processed/olist_orders_abt.csv
  reports/lab02_abt_summary.csv
  reports/lab02_abt_report.md
  figures/lab02_late_delivery_distribution.png
  figures/lab02_review_score_distribution.png
  figures/lab02_payment_value_distribution.png
  figures/lab02_top_customer_states.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent   # project root (parent of src/)
DATA_DIR  = BASE / "data" / "raw"
PROC_DIR  = BASE / "data" / "processed"
REP_DIR   = BASE / "reports"
FIG_DIR   = BASE / "figures"

for d in [PROC_DIR, REP_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 – Load tables
# ══════════════════════════════════════════════════════════════════════════════
print("Loading datasets …")
orders             = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
customers          = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
reviews            = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
payments           = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
order_items        = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
products           = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
category_translation = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")

tables = {
    "orders": orders, "customers": customers, "reviews": reviews,
    "payments": payments, "order_items": order_items,
    "products": products, "category_translation": category_translation,
}
for name, df in tables.items():
    print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 7 – Convert date columns
# ══════════════════════════════════════════════════════════════════════════════
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 8 – Date-based features
# ══════════════════════════════════════════════════════════════════════════════
orders["order_year"]        = orders["order_purchase_timestamp"].dt.year
orders["order_month"]       = orders["order_purchase_timestamp"].dt.month
orders["order_day"]         = orders["order_purchase_timestamp"].dt.day
orders["order_day_of_week"] = orders["order_purchase_timestamp"].dt.dayofweek
orders["order_hour"]        = orders["order_purchase_timestamp"].dt.hour

# ══════════════════════════════════════════════════════════════════════════════
# TASK 9 – Delivery-based features
# ══════════════════════════════════════════════════════════════════════════════
orders["delivery_days"] = (
    orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
).dt.days

orders["estimated_delivery_days"] = (
    orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
).dt.days

orders["delivery_delay_days"] = (
    orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
).dt.days

orders["is_late_delivery"] = np.where(orders["delivery_delay_days"] > 0, 1, 0)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 10 – Late delivery distribution figure
# ══════════════════════════════════════════════════════════════════════════════
print("Generating figure: late_delivery_distribution …")
fig, ax = plt.subplots(figsize=(6, 4))
vc = orders["is_late_delivery"].value_counts().rename(index={0: "On Time", 1: "Late"})
vc.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white", width=0.5)
ax.set_title("Late Delivery Distribution")
ax.set_xlabel("Delivery Status")
ax.set_ylabel("Number of Orders")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height()):,}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "lab02_late_delivery_distribution.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TASK 11-12 – Aggregate payments
# ══════════════════════════════════════════════════════════════════════════════
print("Aggregating payments …")
payment_features = payments.groupby("order_id").agg(
    total_payment_value=("payment_value", "sum"),
    max_payment_installments=("payment_installments", "max"),
    payment_types_count=("payment_type", "nunique"),
).reset_index()

dominant_payment_type = (
    payments.groupby("order_id")["payment_type"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    .reset_index()
    .rename(columns={"payment_type": "dominant_payment_type"})
)
payment_features = payment_features.merge(dominant_payment_type, on="order_id", how="left")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 13-15 – Aggregate order items + product category
# ══════════════════════════════════════════════════════════════════════════════
print("Aggregating order items …")
item_features = order_items.groupby("order_id").agg(
    total_items=("order_item_id", "count"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    unique_products=("product_id", "nunique"),
    unique_sellers=("seller_id", "nunique"),
).reset_index()

items_products = (
    order_items
    .merge(products, on="product_id", how="left")
    .merge(category_translation, on="product_category_name", how="left")
)

order_category = (
    items_products.groupby("order_id")
    .agg(main_product_category=(
        "product_category_name_english",
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    ))
    .reset_index()
)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 16 – Review features
# ══════════════════════════════════════════════════════════════════════════════
print("Aggregating reviews …")
review_features = reviews.groupby("order_id").agg(
    review_score=("review_score", "mean"),
    review_comment_count=("review_comment_message", lambda x: x.notnull().sum()),
    has_review_comment=("review_comment_message", lambda x: int(x.notnull().any())),
).reset_index()

# ══════════════════════════════════════════════════════════════════════════════
# TASK 17 – Build ABT
# ══════════════════════════════════════════════════════════════════════════════
print("Building Analytical Base Table …")
abt = (
    orders
    .merge(customers,        on="customer_id", how="left")
    .merge(review_features,  on="order_id",    how="left")
    .merge(payment_features, on="order_id",    how="left")
    .merge(item_features,    on="order_id",    how="left")
    .merge(order_category,   on="order_id",    how="left")
)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 18 – Target variable
# ══════════════════════════════════════════════════════════════════════════════
abt["is_low_review"] = np.where(abt["review_score"] <= 2, 1, 0)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 19 – Select final columns
# ══════════════════════════════════════════════════════════════════════════════
selected_columns = [
    "order_id", "customer_id", "customer_unique_id",
    "customer_city", "customer_state",
    "order_status",
    "order_year", "order_month", "order_day", "order_day_of_week", "order_hour",
    "delivery_days", "estimated_delivery_days", "delivery_delay_days", "is_late_delivery",
    "review_score", "is_low_review", "review_comment_count", "has_review_comment",
    "total_payment_value", "max_payment_installments", "payment_types_count",
    "dominant_payment_type",
    "total_items", "total_price", "total_freight", "unique_products", "unique_sellers",
    "main_product_category",
]
abt = abt[selected_columns]
print(f"ABT shape: {abt.shape[0]:,} rows × {abt.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 21 – ABT summary report (CSV)
# ══════════════════════════════════════════════════════════════════════════════
print("Saving ABT summary report …")
abt_summary = pd.DataFrame({
    "column":          abt.columns,
    "data_type":       abt.dtypes.astype(str).values,
    "missing_count":   abt.isnull().sum().values,
    "missing_percent": (abt.isnull().sum().values / len(abt) * 100).round(2),
    "unique_values":   [abt[col].nunique() for col in abt.columns],
})
abt_summary.to_csv(REP_DIR / "lab02_abt_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 22 – Visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("Generating figure: review_score_distribution …")
fig, ax = plt.subplots(figsize=(7, 4))
rv = abt["review_score"].dropna()
rv_vc = rv.value_counts().sort_index()
rv_vc.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white", width=0.6)
ax.set_title("Review Score Distribution")
ax.set_xlabel("Review Score")
ax.set_ylabel("Number of Orders")
ax.set_xticklabels([f"{int(v)}" for v in rv_vc.index], rotation=0)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height()):,}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "lab02_review_score_distribution.png")
plt.close()

print("Generating figure: payment_value_distribution …")
fig, ax = plt.subplots(figsize=(8, 4))
abt["total_payment_value"].dropna().hist(bins=60, ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Distribution of Total Payment Value")
ax.set_xlabel("Total Payment Value (BRL)")
ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(FIG_DIR / "lab02_payment_value_distribution.png")
plt.close()

print("Generating figure: top_customer_states …")
fig, ax = plt.subplots(figsize=(8, 4))
top_states = abt["customer_state"].value_counts().head(10)
top_states.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white", width=0.6)
ax.set_title("Top 10 Customer States by Orders")
ax.set_xlabel("Customer State")
ax.set_ylabel("Number of Orders")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for bar in ax.patches:
    ax.annotate(f"{int(bar.get_height()):,}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "lab02_top_customer_states.png")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TASK 23 – Save processed dataset
# ══════════════════════════════════════════════════════════════════════════════
print("Saving processed ABT …")
abt.to_csv(PROC_DIR / "olist_orders_abt.csv", index=False)
assert (PROC_DIR / "olist_orders_abt.csv").exists(), "ERROR: ABT file not saved!"

# ══════════════════════════════════════════════════════════════════════════════
# Generate Markdown report
# ══════════════════════════════════════════════════════════════════════════════
n_orders       = len(abt)
n_cols         = abt.shape[1]
pct_late       = (abt["is_late_delivery"].sum() / n_orders * 100)
pct_low_review = (abt["is_low_review"].sum() / n_orders * 100)
top_state      = abt["customer_state"].value_counts().idxmax()
top_state_n    = abt["customer_state"].value_counts().max()
top_category   = abt["main_product_category"].value_counts().idxmax()
avg_payment    = abt["total_payment_value"].mean()
avg_delivery   = abt["delivery_days"].dropna().mean()
pct_missing_review = (abt["review_score"].isnull().sum() / n_orders * 100)

md = f"""# Lab 2 – Analytical Base Table Report

**Prepared by:** Sharad Laad  
**LinkedIn:** https://www.linkedin.com/in/sharadlaad/  
**Dataset:** Olist Brazilian E-Commerce Public Dataset  
**Lab:** Lab 2 – Building an Analytical Base Table  

---

## 1. ABT Overview

| Metric | Value |
| --- | --- |
| Total Orders (rows) | {n_orders:,} |
| Total Features (columns) | {n_cols} |
| Target Variable | is_low_review |
| Target Type | Binary Classification |

---

## 2. Column Summary

| Column | Data Type | Missing # | Missing % | Unique Values |
| --- | --- | --- | --- | --- |
"""
for _, row in abt_summary.iterrows():
    md += f"| {row['column']} | {row['data_type']} | {int(row['missing_count']):,} | {row['missing_percent']:.1f}% | {int(row['unique_values']):,} |\n"

md += f"""
---

## 3. Key Findings

### 3.1 Delivery Performance

| Metric | Value |
| --- | --- |
| Average Delivery Days | {avg_delivery:.1f} days |
| Late Deliveries | {abt['is_late_delivery'].sum():,} orders ({pct_late:.1f}%) |
| On-Time Deliveries | {(abt['is_late_delivery'] == 0).sum():,} orders ({100-pct_late:.1f}%) |

### 3.2 Target Variable Distribution

| Class | Count | Percentage |
| --- | --- | --- |
| is_low_review = 1 (Low: score ≤ 2) | {int(abt['is_low_review'].sum()):,} | {pct_low_review:.1f}% |
| is_low_review = 0 (High: score > 2) | {int((abt['is_low_review']==0).sum()):,} | {100-pct_low_review:.1f}% |

> **Note:** Orders without a review score ({pct_missing_review:.1f}% of total) are labelled 0 by default
> because `np.where` treats NaN as the "else" branch.

### 3.3 Customer Geography

| Metric | Value |
| --- | --- |
| Top State | {top_state} ({top_state_n:,} orders) |
| Unique States | {abt['customer_state'].nunique()} |
| Unique Cities | {abt['customer_city'].nunique():,} |

### 3.4 Payments

| Metric | Value |
| --- | --- |
| Average Order Value (BRL) | R$ {avg_payment:.2f} |
| Max Order Value (BRL) | R$ {abt['total_payment_value'].max():.2f} |
| Unique Payment Types | {payments['payment_type'].nunique()} |

### 3.5 Products

| Metric | Value |
| --- | --- |
| Top Product Category | {top_category} |
| Unique Categories | {abt['main_product_category'].nunique()} |
| Avg Items per Order | {abt['total_items'].mean():.2f} |

---

## 4. Generated Figures

| Figure | Description |
| --- | --- |
| lab02_late_delivery_distribution.png | Bar chart: on-time vs late deliveries |
| lab02_review_score_distribution.png | Bar chart: frequency of each review score (1–5) |
| lab02_payment_value_distribution.png | Histogram: distribution of total payment value |
| lab02_top_customer_states.png | Bar chart: top 10 Brazilian states by order count |

---

## 5. Reflection Questions

**Q1. What is an Analytical Base Table and why is it important for ML?**  
An ABT is a single, flat, wide table where every row represents one unit of analysis (here, one order) and every column is either a feature or the target variable. It is the direct input to an ML algorithm. Building it correctly — at the right grain, with meaningful features, and a well-defined target — determines the ceiling of model performance.

**Q2. Why do we aggregate payment and order item data before merging?**  
The payments and order_items tables are at a finer grain than orders (multiple rows per order). If we merged them before aggregating, each order would be duplicated for every payment or item row, inflating the dataset and producing incorrect join cardinality.

**Q3. What does it mean for the ABT to be at the "order grain"?**  
It means each row in the ABT represents exactly one order. All features derived from finer-grained tables (payments, items, reviews) have been aggregated (summed, counted, or mode-selected) up to the order level before joining.

**Q4. Why did we define is_low_review as score ≤ 2 instead of score = 1?**  
Score = 1 alone would produce a very small positive class (~3-5%), making the problem extremely imbalanced. Including score = 2 creates a more balanced and business-meaningful class: orders where customers were clearly dissatisfied.

**Q5. What are the risks of leakage if review_score is included as a feature alongside is_low_review?**  
review_score directly encodes the target (is_low_review is derived from it). Including it as a feature would cause severe data leakage: the model would learn a trivial rule (score ≤ 2 → flag = 1) and appear to have 100% accuracy, but would be completely useless in production where the review score is unknown at prediction time.

**Q6. What percentage of orders had a late delivery?**  
Approximately {pct_late:.1f}% of delivered orders arrived after the estimated delivery date. This has significant business implications: late deliveries are a leading predictor of low reviews and customer churn.

**Q7. Which state had the most orders?**  
{top_state} dominated with {top_state_n:,} orders, revealing high geographic concentration in Brazil's southeast. This means ML models may be biased toward patterns from this region.

**Q8. What is the class balance of is_low_review?**  
Approximately {pct_low_review:.1f}% of orders have is_low_review = 1. This moderate class imbalance should be addressed in future labs using techniques like class weighting, oversampling (SMOTE), or threshold tuning during model evaluation.

---

## 6. Output Files

```
data/processed/olist_orders_abt.csv   — Final ABT ({n_orders:,} rows × {n_cols} cols)
reports/lab02_abt_summary.csv         — Column-level metadata
reports/lab02_abt_report.md           — This report
figures/lab02_late_delivery_distribution.png
figures/lab02_review_score_distribution.png
figures/lab02_payment_value_distribution.png
figures/lab02_top_customer_states.png
```
"""

with open(REP_DIR / "lab02_abt_report.md", "w", encoding="utf-8") as f:
    f.write(md)

print("\n[OK] All outputs generated successfully!")
print(f"  ABT: {PROC_DIR / 'olist_orders_abt.csv'}")
print(f"  Summary CSV: {REP_DIR / 'lab02_abt_summary.csv'}")
print(f"  Report MD: {REP_DIR / 'lab02_abt_report.md'}")
print(f"  Figures: {FIG_DIR}")
