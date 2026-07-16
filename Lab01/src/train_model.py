"""
train_model.py
--------------
Trains a Random Forest classifier to predict late delivery
on the Olist dataset and saves the model + metadata.

Usage:
    python src/train_model.py
"""

import sys
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "raw"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
FIGURE_DIR = BASE / "figures"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")

# ─── Build target: Is Delivery Late? ─────────────────────────────────────────
print("Engineering features...")

# Parse timestamps
for col in ["order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date"]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Only keep 'delivered' orders (we know the actual delivery date)
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered = delivered.dropna(subset=["order_delivered_customer_date",
                                     "order_estimated_delivery_date"])

delivered["is_late"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
).astype(int)

delivered["days_to_estimated"] = (
    delivered["order_estimated_delivery_date"]
    - delivered["order_purchase_timestamp"]
).dt.days

delivered["purchase_month"] = delivered["order_purchase_timestamp"].dt.month
delivered["purchase_dow"] = delivered["order_purchase_timestamp"].dt.dayofweek
delivered["purchase_hour"] = delivered["order_purchase_timestamp"].dt.hour

# ─── Add payment features ─────────────────────────────────────────────────────
pay_agg = payments.groupby("order_id").agg(
    total_payment=("payment_value", "sum"),
    n_installments=("payment_installments", "max"),
    n_payment_types=("payment_type", "nunique"),
).reset_index()

# ─── Add order items features ─────────────────────────────────────────────────
items_agg = order_items.groupby("order_id").agg(
    n_items=("order_item_id", "max"),
    total_price=("price", "sum"),
    total_freight=("freight_value", "sum"),
    n_sellers=("seller_id", "nunique"),
).reset_index()

# ─── Add customer state ───────────────────────────────────────────────────────
cust_state = customers[["customer_id", "customer_state"]].copy()
le = LabelEncoder()
cust_state["state_encoded"] = le.fit_transform(cust_state["customer_state"])

# ─── Merge everything ─────────────────────────────────────────────────────────
df = delivered.merge(pay_agg, on="order_id", how="left")
df = df.merge(items_agg, on="order_id", how="left")
df = df.merge(cust_state[["customer_id", "state_encoded"]], on="customer_id", how="left")

# ─── Feature matrix ───────────────────────────────────────────────────────────
FEATURES = [
    "days_to_estimated",
    "purchase_month",
    "purchase_dow",
    "purchase_hour",
    "total_payment",
    "n_installments",
    "n_payment_types",
    "n_items",
    "total_price",
    "total_freight",
    "n_sellers",
    "state_encoded",
]

df = df.dropna(subset=FEATURES + ["is_late"])
X = df[FEATURES].values
y = df["is_late"].values

print(f"Dataset: {len(df)} samples | {y.mean():.2%} late deliveries")

# ─── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─── Train model ──────────────────────────────────────────────────────────────
print("Training Random Forest...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
clf.fit(X_train, y_train)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\nTest Accuracy : {accuracy:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─── Feature Importance Plot ──────────────────────────────────────────────────
importances = clf.feature_importances_
sorted_idx = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(FEATURES)), importances[sorted_idx])
plt.xticks(range(len(FEATURES)), [FEATURES[i] for i in sorted_idx], rotation=45, ha="right")
plt.title("Feature Importances - Late Delivery Prediction")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "model_feature_importances.png", dpi=150)
plt.close()
print(f"\nFeature importance plot saved.")

# ─── Confusion Matrix Plot ────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix - Late Delivery")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i][j]), ha="center", va="center",
                 color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=14)
plt.xticks([0, 1], ["Not Late", "Late"])
plt.yticks([0, 1], ["Not Late", "Late"])
plt.tight_layout()
plt.savefig(FIGURE_DIR / "model_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix plot saved.")

# ─── Save model ───────────────────────────────────────────────────────────────
model_path = MODEL_DIR / "late_delivery_rf.pkl"
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"\nModel saved to {model_path}")

# ─── Save metadata ────────────────────────────────────────────────────────────
metadata = {
    "model_type": "RandomForestClassifier",
    "target": "is_late (1 = late delivery, 0 = on time)",
    "features": FEATURES,
    "n_estimators": 150,
    "max_depth": 12,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "accuracy": round(accuracy, 4),
    "roc_auc": round(roc_auc, 4),
    "classification_report": report,
}

with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model metadata saved.")
print("\nDone!")
