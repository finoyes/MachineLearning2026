"""
pre_push.py
-----------
Generates all expected output files that the plan.md requires to be in the repo:
  - reports/lab01_dataset_summary.csv
  - figures/lab01_order_status_distribution.png
  - figures/lab01_review_score_distribution.png
  - figures/lab01_payment_type_distribution.png
  - figures/lab01_customer_states.png
  - figures/lab01_monthly_orders.png
  - figures/lab01_top_categories.png

Run with the .venv interpreter before pushing.
"""
import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")

DATA_DIR   = Path("data/raw")
REPORT_DIR = Path("reports")
FIG_DIR    = Path("figures")
REPORT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

print("Loading datasets...")
customers         = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
geolocation       = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
order_items       = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
payments          = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
reviews           = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")
orders            = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
products          = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
sellers           = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
category_translation = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")

tables = {
    "customers": customers, "geolocation": geolocation,
    "order_items": order_items, "payments": payments,
    "reviews": reviews, "orders": orders,
    "products": products, "sellers": sellers,
    "category_translation": category_translation,
}

# ── Dataset Summary CSV ────────────────────────────────────────────────────────
print("Generating reports/lab01_dataset_summary.csv ...")
summary = pd.DataFrame([{
    "table_name":    name,
    "rows":          df.shape[0],
    "columns":       df.shape[1],
    "missing_values": df.isnull().sum().sum(),
    "duplicate_rows": df.duplicated().sum(),
} for name, df in tables.items()])
summary.to_csv(REPORT_DIR / "lab01_dataset_summary.csv", index=False)
print(summary.to_string(index=False))

# ── Helper ────────────────────────────────────────────────────────────────────
def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150)
    plt.close()
    print(f"  Saved figures/{name}")

# ── Order Status ──────────────────────────────────────────────────────────────
print("\nGenerating figures ...")
fig, ax = plt.subplots(figsize=(9, 5))
orders["order_status"].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Order Status Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Order Status"); ax.set_ylabel("Number of Orders")
ax.tick_params(axis="x", rotation=35)
save("lab01_order_status_distribution.png")

# ── Review Score ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
reviews["review_score"].value_counts().sort_index().plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="white")
ax.set_title("Review Score Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Review Score (1=Worst, 5=Best)"); ax.set_ylabel("Number of Reviews")
ax.tick_params(axis="x", rotation=0)
save("lab01_review_score_distribution.png")

# ── Payment Type ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
payments["payment_type"].value_counts().plot(kind="bar", ax=ax, color="coral", edgecolor="white")
ax.set_title("Payment Type Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Payment Type"); ax.set_ylabel("Number of Payments")
ax.tick_params(axis="x", rotation=20)
save("lab01_payment_type_distribution.png")

# ── Customer States ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
customers["customer_state"].value_counts().head(10).plot(kind="bar", ax=ax, color="slateblue", edgecolor="white")
ax.set_title("Top 10 Customer States", fontsize=14, fontweight="bold")
ax.set_xlabel("State"); ax.set_ylabel("Number of Customers")
ax.tick_params(axis="x", rotation=0)
save("lab01_customer_states.png")

# ── Monthly Orders ────────────────────────────────────────────────────────────
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["year_month"] = orders["order_purchase_timestamp"].dt.to_period("M")
monthly = orders.groupby("year_month").size()
fig, ax = plt.subplots(figsize=(14, 5))
monthly.plot(ax=ax, marker="o", color="steelblue")
ax.set_title("Monthly Order Count", fontsize=14, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Number of Orders")
ax.tick_params(axis="x", rotation=45)
save("lab01_monthly_orders.png")

# ── Top Product Categories ────────────────────────────────────────────────────
top_cats = (
    order_items
    .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    .groupby("product_category_name")["order_id"]
    .nunique()
    .sort_values(ascending=False)
    .head(10)
)
fig, ax = plt.subplots(figsize=(10, 5))
top_cats.plot(kind="bar", ax=ax, color="darkorange", edgecolor="white")
ax.set_title("Top 10 Product Categories by Orders", fontsize=14, fontweight="bold")
ax.set_xlabel("Category"); ax.set_ylabel("Number of Orders")
ax.tick_params(axis="x", rotation=40)
save("lab01_top_categories.png")

print("\nAll outputs generated successfully!")
print("\nFiles ready to commit:")
for f in sorted(FIG_DIR.glob("*.png")):
    print(f"  figures/{f.name}")
for f in sorted(REPORT_DIR.glob("*")):
    print(f"  reports/{f.name}")
