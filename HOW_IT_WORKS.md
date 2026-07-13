# How It Works — Model & Program Explanation

> **Prepared by:** Sharad Laad | ORY AI Labs
> **LinkedIn:** https://www.linkedin.com/in/sharadlaad/
> **Course:** Machine Learning — Lab 1

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Folder Structure](#2-project-folder-structure)
3. [The Dataset — Olist E-Commerce](#3-the-dataset--olist-e-commerce)
   - 3.1 [Tables and Schemas](#31-tables-and-schemas)
   - 3.2 [Relational Map](#32-relational-map)
4. [The Machine Learning Problem](#4-the-machine-learning-problem)
5. [Program Walkthrough — `train_model.py`](#5-program-walkthrough--train_modelpy)
   - 5.1 [Step 1 — Load Raw Data](#51-step-1--load-raw-data)
   - 5.2 [Step 2 — Build the Target Variable](#52-step-2--build-the-target-variable)
   - 5.3 [Step 3 — Feature Engineering](#53-step-3--feature-engineering)
   - 5.4 [Step 4 — Merge Into One Training Table](#54-step-4--merge-into-one-training-table)
   - 5.5 [Step 5 — Train / Test Split](#55-step-5--train--test-split)
   - 5.6 [Step 6 — Train the Model](#56-step-6--train-the-model)
   - 5.7 [Step 7 — Evaluate](#57-step-7--evaluate)
   - 5.8 [Step 8 — Save Artifacts](#58-step-8--save-artifacts)
6. [The Model — Random Forest Classifier](#6-the-model--random-forest-classifier)
   - 6.1 [What is a Decision Tree?](#61-what-is-a-decision-tree)
   - 6.2 [What is a Random Forest?](#62-what-is-a-random-forest)
   - 6.3 [Hyperparameters Used](#63-hyperparameters-used)
   - 6.4 [Why Random Forest for This Problem?](#64-why-random-forest-for-this-problem)
7. [Features Explained](#7-features-explained)
8. [Model Performance](#8-model-performance)
   - 8.1 [Metrics Explained](#81-metrics-explained)
   - 8.2 [Results](#82-results)
   - 8.3 [Understanding the Confusion Matrix](#83-understanding-the-confusion-matrix)
   - 8.4 [Class Imbalance Problem](#84-class-imbalance-problem)
9. [Outputs Generated](#9-outputs-generated)
10. [The Jupyter Notebook — Lab01](#10-the-jupyter-notebook--lab01)
11. [How to Run Everything](#11-how-to-run-everything)
12. [Concepts Glossary](#12-concepts-glossary)

---

## 1. Project Overview

This project is a complete end-to-end Machine Learning workflow built on the
**Olist Brazilian E-Commerce Public Dataset**. It covers every phase of a
real-world ML project:

```
Business Problem
     ↓
Data Loading & Inspection
     ↓
Exploratory Data Analysis (EDA)
     ↓
Feature Engineering
     ↓
Model Training (Random Forest)
     ↓
Evaluation & Visualisation
     ↓
Saving the Model
```

The **business question** being answered is:

> *"Given information available at the time of purchase, can we predict whether
> a customer's order will arrive later than the estimated delivery date?"*

---

## 2. Project Folder Structure

```
MachineLearning2026/
│
├── data/
│   ├── raw/                        ← Original CSV files (9 files, never modified)
│   └── processed/                  ← Cleaned data (created in later labs)
│
├── notebooks/
│   └── Lab01_Olist_Dataset_Introduction.ipynb   ← 81-cell EDA notebook
│
├── src/
│   └── train_model.py              ← Full ML pipeline script
│
├── models/
│   ├── late_delivery_rf.pkl        ← Serialised trained model (pickle)
│   └── model_metadata.json         ← Model stats, features, performance
│
├── figures/
│   ├── model_feature_importances.png
│   ├── model_confusion_matrix.png
│   ├── lab01_order_status_distribution.png
│   └── ...
│
├── reports/
│   ├── lab01_dataset_summary.csv
│   └── lab01_data_dictionary.md
│
├── .gitignore                      ← Excludes data, models, venv from Git
├── requirements.txt                ← Python dependencies
└── README.md                       ← Project description
```

> [!IMPORTANT]
> `data/raw/` and `models/` are listed in `.gitignore` and will **not** be
> pushed to GitHub. This is intentional — raw data files are large and
> trained model binaries should be tracked with a dedicated tool like MLflow.

---

## 3. The Dataset — Olist E-Commerce

The Olist dataset is a **relational dataset** consisting of 9 CSV files that
represent the full lifecycle of an e-commerce order in Brazil, from customer
registration through to delivery and review.

### 3.1 Tables and Schemas

| Table | Rows | Key Columns |
|---|---|---|
| `olist_customers_dataset.csv` | 99,441 | `customer_id`, `customer_unique_id`, `customer_state` |
| `olist_orders_dataset.csv` | 99,441 | `order_id`, `customer_id`, `order_status`, delivery timestamps |
| `olist_order_items_dataset.csv` | ~147,000 | `order_id`, `product_id`, `seller_id`, `price`, `freight_value` |
| `olist_order_payments_dataset.csv` | ~113,000 | `order_id`, `payment_type`, `payment_value`, `payment_installments` |
| `olist_order_reviews_dataset.csv` | ~79,000 | `order_id`, `review_score`, `review_comment_message` |
| `olist_products_dataset.csv` | 32,951 | `product_id`, `product_category_name`, physical dimensions |
| `olist_sellers_dataset.csv` | 3,095 | `seller_id`, `seller_state` |
| `olist_geolocation_dataset.csv` | 1,000,000 | `geolocation_zip_code_prefix`, lat/lng |
| `product_category_name_translation.csv` | 20 | Portuguese → English category name |

### 3.2 Relational Map

```
customers ──(customer_id)──► orders
                               │
                   ┌───────────┼───────────────┐
                   ▼           ▼               ▼
             order_items    payments        reviews
                   │
           ┌───────┴──────┐
           ▼              ▼
        products       sellers
           │
           ▼
    category_translation
```

**Key Joins:**
- `orders` ↔ `customers` — via `customer_id`
- `orders` ↔ `order_items` — via `order_id`
- `orders` ↔ `payments` — via `order_id`
- `orders` ↔ `reviews` — via `order_id`
- `order_items` ↔ `products` — via `product_id`
- `order_items` ↔ `sellers` — via `seller_id`

---

## 4. The Machine Learning Problem

**Problem Type:** Binary Classification

**Target Variable:**
```
is_late = 1  →  Order arrived AFTER the estimated delivery date (Late)
is_late = 0  →  Order arrived ON or BEFORE the estimated date (On Time)
```

**Why this problem?**
- Late deliveries directly hurt customer satisfaction (reflected in low review scores).
- If a company can predict that an order will be late *at the time of purchase*, it
  can proactively notify the customer, offer compensation, or re-route the shipment.
- All the information we use as features is available **at the time the order is placed**,
  making this a realistic production ML problem.

**Class Distribution:**
```
Class 0 (On Time)  →  ~95.2% of orders
Class 1 (Late)     →  ~4.8% of orders
```

This is a **class imbalance** problem. Only ~5% of orders are late, so a naive model
that always predicts "on time" would be 95% accurate but completely useless. This is
why we use `class_weight="balanced"` in the Random Forest — explained in Section 6.3.

---

## 5. Program Walkthrough — `train_model.py`

File: [`src/train_model.py`](src/train_model.py)

### 5.1 Step 1 — Load Raw Data

```python
orders      = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
customers   = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
payments    = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
```

Only 4 of the 9 tables are needed for this model. The others (geolocation, reviews,
products, sellers) would be used in more advanced versions of the model.

The `BASE` path is resolved dynamically from the script's own file location so it
works regardless of where you run it from:

```python
BASE = Path(__file__).resolve().parent.parent  # Goes up from src/ to project root
```

---

### 5.2 Step 2 — Build the Target Variable

```python
# Convert timestamp strings to proper Python datetime objects
for col in ["order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Only use DELIVERED orders — we only know if it was late for delivered orders
delivered = orders[orders["order_status"] == "delivered"].copy()
delivered = delivered.dropna(subset=["order_delivered_customer_date",
                                     "order_estimated_delivery_date"])

# Build target: 1 if actual delivery is after estimated delivery
delivered["is_late"] = (
    delivered["order_delivered_customer_date"]
    > delivered["order_estimated_delivery_date"]
).astype(int)
```

**Why filter to `delivered` only?**
Orders with status `canceled`, `processing`, `invoiced`, etc. do not have a
`order_delivered_customer_date`, so we cannot determine whether they were late or
not. Filtering ensures we only train on rows with a definitive ground truth.

---

### 5.3 Step 3 — Feature Engineering

Feature engineering is the process of extracting useful signals from raw data.

#### Time-based features (from purchase timestamp)

```python
delivered["days_to_estimated"] = (
    estimated_delivery_date - order_purchase_timestamp
).dt.days

delivered["purchase_month"] = order_purchase_timestamp.dt.month   # 1–12
delivered["purchase_dow"]   = order_purchase_timestamp.dt.dayofweek  # 0=Mon, 6=Sun
delivered["purchase_hour"]  = order_purchase_timestamp.dt.hour    # 0–23
```

- **`days_to_estimated`** — How many days between purchase and estimated delivery.
  Longer windows give sellers more time to deliver on time.
- **`purchase_month`** — Demand spikes in certain months (e.g. Black Friday November)
  may strain logistics and cause more late deliveries.
- **`purchase_dow`** — Orders placed on Fridays may not be dispatched until Monday.
- **`purchase_hour`** — Late-night orders may miss the next-day dispatch cutoff.

#### Payment features (aggregated per order)

```python
pay_agg = payments.groupby("order_id").agg(
    total_payment   = ("payment_value",       "sum"),     # Total amount paid
    n_installments  = ("payment_installments", "max"),    # Max instalment plan
    n_payment_types = ("payment_type",         "nunique") # Num payment methods used
)
```

- **`total_payment`** — A proxy for order value. Higher-value orders may have
  different fulfilment behaviour.
- **`n_installments`** — More instalments suggest a larger or more complex order.
- **`n_payment_types`** — Using multiple payment methods (e.g. voucher + credit card)
  may indicate a complex checkout.

#### Order item features (aggregated per order)

```python
items_agg = order_items.groupby("order_id").agg(
    n_items       = ("order_item_id",   "max"),      # Number of items in order
    total_price   = ("price",           "sum"),      # Total product cost
    total_freight = ("freight_value",   "sum"),      # Total shipping cost
    n_sellers     = ("seller_id",       "nunique")   # Sellers involved
)
```

- **`n_items`** — Multi-item orders have more things to pack and can be harder to ship.
- **`total_freight`** — Higher freight may mean the item is going far (harder to deliver on time).
- **`n_sellers`** — Orders spanning multiple sellers must all be ready before dispatch.

#### Customer location feature

```python
le = LabelEncoder()
cust_state["state_encoded"] = le.fit_transform(cust_state["customer_state"])
```

`LabelEncoder` converts state strings (`"SP"`, `"RJ"`, `"MG"`, ...) to integers
(`0`, `1`, `2`, ...) because machine learning models work with numbers, not strings.
Some states are further from major distribution centres and therefore have higher
late delivery rates.

---

### 5.4 Step 4 — Merge Into One Training Table

```python
df = delivered.merge(pay_agg,   on="order_id",   how="left")
df = df.merge(items_agg,        on="order_id",   how="left")
df = df.merge(cust_state[...],  on="customer_id", how="left")
```

A **left merge** keeps all rows from the left table (`delivered`) and fills in
matching columns from the right table. If an order has no payment record (rare),
the payment columns will be `NaN`. These rows are dropped before training.

---

### 5.5 Step 5 — Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 20% held out for testing
    random_state=42,     # Fixed seed for reproducibility
    stratify=y           # Maintain class proportions in both splits
)
```

| Split | Samples | Purpose |
|---|---|---|
| Training set (80%) | 76,885 | Model learns patterns from this |
| Test set (20%) | 19,222 | Model evaluated on this (never seen during training) |

**Why `stratify=y`?**
Because only ~5% of orders are late, random splitting might accidentally put very
few late orders in the test set. Stratification ensures both splits contain ~5%
late orders, making the evaluation fair and representative.

---

### 5.6 Step 6 — Train the Model

```python
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
clf.fit(X_train, y_train)
```

This fits 150 decision trees to the training data. See Section 6 for a full
explanation of the algorithm and each hyperparameter.

---

### 5.7 Step 7 — Evaluate

```python
y_pred = clf.predict(X_test)           # Hard predictions: 0 or 1
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of being Late

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)
```

Two plots are saved to `figures/`:

1. **Feature Importance Bar Chart** — ranks features by how much they contributed
   to reducing prediction error across all 150 trees.
2. **Confusion Matrix** — shows exactly how many orders were correctly and
   incorrectly classified in each category.

---

### 5.8 Step 8 — Save Artifacts

```python
# Save the trained model as a binary pickle file
with open("models/late_delivery_rf.pkl", "wb") as f:
    pickle.dump(clf, f)

# Save performance stats as JSON for logging / reporting
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

**Pickle (`.pkl`)** serialises the Python object into bytes so it can be loaded
back later without retraining:

```python
import pickle
with open("models/late_delivery_rf.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict([[30, 11, 4, 15, 250.0, 3, 1, 2, 180.0, 35.0, 1, 5]])
print("Late?" , "Yes" if prediction[0] == 1 else "No")
```

---

## 6. The Model — Random Forest Classifier

### 6.1 What is a Decision Tree?

A **Decision Tree** makes predictions by asking a series of yes/no questions
about the input features, splitting the data at each node, until it reaches a
leaf node with a prediction.

```
Is days_to_estimated < 20?
        ├── YES → Is total_freight > 40?
        │           ├── YES → Predict: LATE (1)
        │           └── NO  → Predict: ON TIME (0)
        └── NO  → Is purchase_month == 11?
                    ├── YES → Predict: LATE (1)
                    └── NO  → Predict: ON TIME (0)
```

A single deep tree tends to **overfit** — it memorises the training data
perfectly but performs poorly on new data.

### 6.2 What is a Random Forest?

A **Random Forest** fixes overfitting by building many different trees and
combining their predictions (majority vote for classification).

Two sources of randomness make each tree different:

1. **Bootstrap Sampling (Bagging):** Each tree trains on a different random
   sample (with replacement) of the training data. Some rows appear multiple
   times, others not at all.

2. **Random Feature Selection:** At each split in the tree, only a random
   *subset* of features is considered. This prevents all trees from always
   splitting on the same dominant feature.

```
Training Data
     │
     ├──[Bootstrap Sample 1]── Tree 1 ── Prediction 1
     ├──[Bootstrap Sample 2]── Tree 2 ── Prediction 2
     ├──[Bootstrap Sample 3]── Tree 3 ── Prediction 3
     │          ...
     └──[Bootstrap Sample 150]─ Tree 150 ─ Prediction 150
                                                 │
                                         Majority Vote
                                                 │
                                       Final Prediction
```

Because each tree sees different data and different features, they make
different errors. When combined, errors cancel out and accuracy improves.

### 6.3 Hyperparameters Used

| Hyperparameter | Value | What it Controls |
|---|---|---|
| `n_estimators` | `150` | Number of trees in the forest. More trees = more stable, but slower. |
| `max_depth` | `12` | Maximum depth of each tree. Limits overfitting. `None` = fully grown. |
| `min_samples_leaf` | `5` | Minimum number of training samples required at a leaf node. Prevents tiny, overfit leaves. |
| `random_state` | `42` | Fixed seed for reproducibility — same result every time you run it. |
| `n_jobs` | `-1` | Use all available CPU cores for parallel training. |
| `class_weight` | `"balanced"` | Automatically adjusts weights inversely proportional to class frequency. Makes the model pay more attention to the rare `is_late=1` class. |

### 6.4 Why Random Forest for This Problem?

| Reason | Explanation |
|---|---|
| **Handles mixed features** | Works well with both numerical and label-encoded categorical data without scaling. |
| **Robust to outliers** | Tree splits are based on rank/order, not raw values. |
| **Feature importance** | Provides a built-in ranking of which features matter most. |
| **No scaling needed** | Unlike SVMs or logistic regression, Random Forests don't require `StandardScaler`. |
| **Works with imbalanced data** | `class_weight="balanced"` handles the ~5% late delivery rate naturally. |
| **Strong baseline** | Consistently achieves good results with minimal tuning, making it ideal for Lab 1. |

---

## 7. Features Explained

The model uses **12 features**, all derived from data available at the time of
purchase (no data leakage):

| # | Feature | Source Table | Description | Intuition |
|---|---|---|---|---|
| 1 | `days_to_estimated` | `orders` | Days between purchase and estimated delivery | More buffer time → less likely to be late |
| 2 | `purchase_month` | `orders` | Month of purchase (1–12) | November (Black Friday) has more late deliveries |
| 3 | `purchase_dow` | `orders` | Day of week (0=Mon, 6=Sun) | Weekend orders may miss dispatch cutoff |
| 4 | `purchase_hour` | `orders` | Hour of day (0–23) | Late-night orders often dispatched next day |
| 5 | `total_payment` | `payments` | Total amount paid (BRL) | Proxy for order value and complexity |
| 6 | `n_installments` | `payments` | Maximum instalment plan chosen | Larger instalments suggest larger purchases |
| 7 | `n_payment_types` | `payments` | Number of payment methods used | Complex payment might signal complex order |
| 8 | `n_items` | `order_items` | Number of items in the order | Multi-item orders are harder to fulfil |
| 9 | `total_price` | `order_items` | Sum of all item prices | Proxy for basket size |
| 10 | `total_freight` | `order_items` | Sum of all freight costs | Higher freight = longer distance |
| 11 | `n_sellers` | `order_items` | Number of unique sellers in the order | Multi-seller orders depend on multiple suppliers |
| 12 | `state_encoded` | `customers` | Integer code for customer state | Some states are harder to reach on time |

> [!NOTE]
> Features 2–4 (month, day-of-week, hour) are called **temporal features**.
> They capture the effect of *when* the order was placed on delivery performance.

---

## 8. Model Performance

### 8.1 Metrics Explained

#### Accuracy
The proportion of all predictions that were correct.
```
Accuracy = (True Positives + True Negatives) / Total Predictions
         = 0.8552  →  85.52%
```
This sounds high, but is misleading here due to class imbalance. See Section 8.4.

#### ROC-AUC (Receiver Operating Characteristic — Area Under Curve)
Measures how well the model separates the two classes across all possible
classification thresholds. Ranges from 0.5 (random guessing) to 1.0 (perfect).
```
ROC-AUC = 0.9296  →  Very strong separation ability
```

#### Precision
Of all orders the model *predicted* as late, what fraction actually were late?
```
Precision (Late) = 0.24  →  24% of "late" predictions were correct
```

#### Recall
Of all orders that *actually were* late, what fraction did the model catch?
```
Recall (Late) = 0.92  →  The model caught 92% of all late deliveries
```

#### F1-Score
The harmonic mean of Precision and Recall. Useful when classes are imbalanced.
```
F1 (Late) = 2 × (Precision × Recall) / (Precision + Recall) = 0.38
```

### 8.2 Results

| Metric | Value |
|---|---|
| Test Accuracy | **85.52%** |
| ROC-AUC | **0.9296** |
| Precision (Late=1) | 0.24 |
| Recall (Late=1) | **0.92** |
| F1-Score (Late=1) | 0.38 |
| Training Samples | 76,885 |
| Test Samples | 19,222 |

### 8.3 Understanding the Confusion Matrix

```
                Predicted: On Time    Predicted: Late
Actual: On Time      15,557              2,735        ← False Positives (Type I Error)
Actual: Late            73                857         ← True Positives (correctly caught)
```

- **True Negatives (15,557):** Correctly predicted as on time — good.
- **True Positives (857):** Late orders correctly identified — good.
- **False Positives (2,735):** On-time orders wrongly flagged as late — the model over-alerts.
- **False Negatives (73):** Late orders missed by the model — these sneak through undetected.

**Trade-off:** Because we use `class_weight="balanced"`, the model is tuned to
catch as many late deliveries as possible (high recall = 92%), at the cost of also
flagging some on-time orders (lower precision = 24%). For a business, missing a
late delivery is more costly than a false alarm, so high recall is preferred.

### 8.4 Class Imbalance Problem

Our dataset has ~95% on-time and ~5% late orders. Without correction, the model
would learn to always say "on time" and achieve 95% accuracy while being completely
useless for detecting late deliveries.

**Solution used:** `class_weight="balanced"` in `RandomForestClassifier`.

This internally multiplies the loss for the minority class (late) by
`(n_samples / (n_classes × n_samples_in_class))`, making errors on minority class
samples count more during training.

---

## 9. Outputs Generated

After running `python src/train_model.py`, the following files are created:

| Output File | Description |
|---|---|
| `models/late_delivery_rf.pkl` | Serialised trained Random Forest model |
| `models/model_metadata.json` | JSON with accuracy, ROC-AUC, features, hyperparams |
| `figures/model_feature_importances.png` | Bar chart of feature importances |
| `figures/model_confusion_matrix.png` | Confusion matrix heatmap |

---

## 10. The Jupyter Notebook — Lab01

File: [`notebooks/Lab01_Olist_Dataset_Introduction.ipynb`](notebooks/Lab01_Olist_Dataset_Introduction.ipynb)

The notebook has **81 cells** and covers all 21 tasks from the lab manual:

| Tasks | Content |
|---|---|
| Tasks 3–4 | Import libraries, define dataset path |
| Task 5 | Load all 9 CSV files |
| Task 6 | Display shape (rows × columns) of each table |
| Task 7 | Preview first 5 rows of each table (`head()`) |
| Task 8 | Inspect data types for all tables (`info()`) |
| Task 9 | Generate a summary table (rows, columns, missing values, duplicates) |
| Task 10 | Missing value analysis per table |
| Task 11 | Duplicate record analysis |
| Task 12 | Explore orders table + bar chart |
| Task 13 | Explore reviews table + bar chart |
| Task 14 | Explore payments table + bar chart |
| Task 15 | Explore products table |
| Task 16 | Explore customers table + top states chart |
| Task 17 | Answer 6 business questions with Pandas |
| Task 18 | Identify 7 potential ML problems from the dataset |
| Task 19 | Create data dictionary for 3 tables |
| Tasks 20–21 | README and Git/GitHub setup instructions |
| Extension | Merge tables, monthly order trend, top categories chart |
| Reflection | Full answers to all 10 reflection questions |

---

## 11. How to Run Everything

### Step 1 — Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Place Dataset Files

Place all 9 Olist CSV files inside `data/raw/`. If you are using the synthetic
dataset generated by `generate_dataset.py`, they are already there.

### Step 4 — Run the Model Training Script

```bash
python src/train_model.py
```

### Step 5 — Open the Jupyter Notebook

```bash
jupyter notebook notebooks/Lab01_Olist_Dataset_Introduction.ipynb
```

### Step 6 — Load the Trained Model in a New Script

```python
import pickle

with open("models/late_delivery_rf.pkl", "rb") as f:
    model = pickle.load(f)

# Feature order: [days_to_estimated, purchase_month, purchase_dow, purchase_hour,
#                 total_payment, n_installments, n_payment_types,
#                 n_items, total_price, total_freight, n_sellers, state_encoded]
sample = [[30, 11, 4, 23, 350.0, 6, 1, 3, 280.0, 62.0, 2, 17]]
prediction = model.predict(sample)
probability = model.predict_proba(sample)[0][1]

print(f"Will it be late? {'YES' if prediction[0] == 1 else 'NO'}")
print(f"Probability of being late: {probability:.2%}")
```

---

## 12. Concepts Glossary

| Term | Definition |
|---|---|
| **Classification** | Predicting which category/class an input belongs to |
| **Binary Classification** | Classification with exactly two possible outcomes |
| **Feature** | An input variable used by the model to make predictions |
| **Feature Engineering** | Creating new, informative features from raw data |
| **Target Variable** | The output the model is trying to predict (`is_late`) |
| **Training Set** | Data used to teach the model |
| **Test Set** | Data held out and never seen during training; used to evaluate |
| **Overfitting** | When a model learns training data too well and fails on new data |
| **Decision Tree** | A tree-structured model making decisions via if/else rules |
| **Random Forest** | An ensemble of many decision trees voting on a final answer |
| **Bootstrap Sampling** | Random sampling with replacement to create diverse training sets |
| **Bagging** | Combining many models trained on different bootstrap samples |
| **Class Imbalance** | When one class appears far more frequently than another |
| **Class Weights** | Penalties applied to correct for class imbalance during training |
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of predicted positives, how many were truly positive |
| **Recall** | Of true positives, how many did the model correctly identify |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the ROC curve; measures classification ability across thresholds |
| **Confusion Matrix** | Table showing TP, TN, FP, FN counts |
| **Pickle** | Python serialisation format for saving objects to disk |
| **Label Encoding** | Converting categorical text labels to integer codes |
| **Left Merge** | SQL-style join that keeps all rows from the left table |
| **`stratify=y`** | Ensures both train and test splits have the same class proportions |
| **`n_jobs=-1`** | Use all CPU cores available for parallel computation |

---

*End of Document — Machine Learning Engineering Studio 1*

*Prepared by Sharad Laad | ORY AI Labs*
*LinkedIn: https://www.linkedin.com/in/sharadlaad/*
