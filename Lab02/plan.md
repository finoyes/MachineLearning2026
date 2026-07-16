# Lab 2: Building an Analytical Base Table (ABT)

**Prepared by:** Sharad Laad  
**LinkedIn:** https://www.linkedin.com/in/sharadlaad/  
**Course:** Machine Learning  
**Unit:** Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering  
**Duration:** 2 Hours  
**Dataset:** Olist Brazilian E-Commerce Public Dataset  

---

## 1. Experiment Information

| Item | Details |
| --- | --- |
| Experiment No. | 2 |
| Experiment Title | Building an Analytical Base Table for Machine Learning |
| Course | Machine Learning |
| Unit | Unit 1 – Mathematical Foundations, Data Engineering & Feature Engineering |
| Duration | 2 Hours |
| Dataset | Olist Brazilian E-Commerce Public Dataset |
| Programming Language | Python |
| Recommended IDE | VS Code or Cursor |
| Notebook Environment | Jupyter Notebook |
| Main Libraries | Pandas, NumPy, Matplotlib, Seaborn |
| Submission Type | Jupyter Notebook + Reports + Figures + Processed Data |

---

## 2. Purpose of This Laboratory

An Analytical Base Table (ABT) is the foundational dataset used to train a Machine Learning model. It is a **single, flat, wide table** where each row represents one observation (in our case, one order) and each column is either a feature (input) or a target variable (output).

Building the ABT is the most critical data engineering step. It requires:
- Merging multiple relational tables into a single flat structure
- Creating new features from existing columns (feature engineering)
- Aggregating transactional data to the correct grain
- Defining the target variable (what we want to predict)
- Validating the shape and quality of the resulting table

---

## 3. Learning Objectives

After completing this laboratory, students will be able to:

1. Understand the concept of an Analytical Base Table and its role in ML pipelines.
2. Merge multiple relational tables into a single analytical dataset.
3. Engineer date-based and delivery-based features from raw timestamps.
4. Aggregate transactional tables (payments, order items) to the order grain.
5. Define a binary target variable for classification.
6. Generate a structured ABT summary report.
7. Visualize key feature distributions for exploratory validation.
8. Save a production-ready processed dataset for use in future labs.

---

## 4. Learning Philosophy

This course follows the principle:

> **Business Problem → Data Understanding → Data Cleaning → Feature Engineering → Model Building → Evaluation → Deployment**

Lab 2 sits at the **Feature Engineering** stage. The ABT produced here will be the direct input for ML models in later labs.

---

## 5. Business Context

The Olist Brazilian E-Commerce dataset contains transactional data from a marketplace platform. The central business question for this lab is:

> **"Can we predict whether a customer will leave a low review (score <= 2) based on their order characteristics?"**

This is a binary classification problem:
- **Target Variable:** `is_low_review` (1 = review score <= 2, 0 = review score > 2)
- **Features:** Order metadata, payment details, delivery timing, product category, customer location

---

## 6. Dataset Schema — Table Relationships

```
customers (customer_id)
      |
      +-- orders (customer_id -> order_id)
                |
                +-- order_items (order_id -> product_id, seller_id)
                |         +-- products (product_id -> category)
                |                   +-- category_translation
                +-- payments (order_id)
                +-- reviews (order_id)
```

| Table | Grain | Key |
| --- | --- | --- |
| orders | One row per order | order_id |
| customers | One row per order-customer | customer_id |
| order_items | One or more rows per order | order_id |
| payments | One or more rows per order | order_id |
| reviews | Usually one row per order | order_id |
| products | One row per product | product_id |
| category_translation | One row per category | product_category_name |

---

## 7. ABT Design Plan

### 7.1 Target Variable

| Variable | Type | Definition |
| --- | --- | --- |
| is_low_review | Binary (0/1) | 1 if review_score <= 2, else 0 |

### 7.2 Feature Groups

#### Group A: Order Time Features (from orders)
| Feature | Source | Engineering |
| --- | --- | --- |
| order_year | order_purchase_timestamp | dt.year |
| order_month | order_purchase_timestamp | dt.month |
| order_day | order_purchase_timestamp | dt.day |
| order_day_of_week | order_purchase_timestamp | dt.dayofweek |
| order_hour | order_purchase_timestamp | dt.hour |

#### Group B: Delivery Features (from orders)
| Feature | Source | Engineering |
| --- | --- | --- |
| delivery_days | delivered_date - purchase_date | timedelta.days |
| estimated_delivery_days | estimated_date - purchase_date | timedelta.days |
| delivery_delay_days | delivered_date - estimated_date | timedelta.days |
| is_late_delivery | delivery_delay_days > 0 | Binary flag |

#### Group C: Payment Features (aggregated from payments)
| Feature | Aggregation |
| --- | --- |
| total_payment_value | sum(payment_value) per order |
| max_payment_installments | max(payment_installments) per order |
| payment_types_count | nunique(payment_type) per order |
| dominant_payment_type | mode(payment_type) per order |

#### Group D: Item Features (aggregated from order_items + products)
| Feature | Aggregation |
| --- | --- |
| total_items | count(order_item_id) per order |
| total_price | sum(price) per order |
| total_freight | sum(freight_value) per order |
| unique_products | nunique(product_id) per order |
| unique_sellers | nunique(seller_id) per order |
| main_product_category | mode(product_category_name_english) per order |

#### Group E: Review Features (aggregated from reviews)
| Feature | Aggregation |
| --- | --- |
| review_score | mean(review_score) per order |
| review_comment_count | count of non-null review_comment_message |
| has_review_comment | 1 if any non-null comment, else 0 |

#### Group F: Customer Features (from customers)
| Feature | Source |
| --- | --- |
| customer_unique_id | customers table |
| customer_city | customers table |
| customer_state | customers table |

---

## 8. Tasks

### Task 1: Review Lab Manual (Lab02_Building_Analytical_Base_Table_Manual)

### Task 2: Import Required Libraries
- pandas, numpy, matplotlib, seaborn
- pathlib.Path

### Task 3: Define Folder Paths
- DATA_DIR = ../data/raw
- PROCESSED_DIR = ../data/processed
- REPORTS_DIR = ../reports
- FIGURES_DIR = ../figures

### Task 4: Load Required Tables
- orders, customers, reviews, payments, order_items, products, category_translation

### Task 5: Inspect Key Tables
- Use .head() on each loaded table

### Task 6: Understand the Level of Each Table
- Document the grain (row-level) of each table

### Task 7: Convert Date Columns
- Convert all timestamp columns in orders to datetime using pd.to_datetime

### Task 8: Create Date-Based Features
- Extract year, month, day, day_of_week, hour from order_purchase_timestamp

### Task 9: Create Delivery-Based Features
- Calculate delivery_days, estimated_delivery_days, delivery_delay_days
- Create binary flag is_late_delivery

### Task 10: Analyze Late Delivery Distribution
- value_counts() on is_late_delivery
- Bar chart -> save to figures/lab02_late_delivery_distribution.png

### Task 11: Aggregate Payments Data
- Group by order_id: sum payment_value, max installments, nunique payment_type

### Task 12: Add Dominant Payment Type
- mode of payment_type per order_id

### Task 13: Aggregate Order Items Data
- Group by order_id: count items, sum price, sum freight, nunique products/sellers

### Task 14: Add Product Category Information
- Join order_items -> products -> category_translation

### Task 15: Create Main Product Category Per Order
- mode of product_category_name_english per order_id

### Task 16: Prepare Review Features
- Group by order_id: mean score, count comments, has_comment flag

### Task 17: Create the Analytical Base Table
- Merge: orders + customers + reviews + payments + items + category

### Task 18: Create Low Review Target Variable
- is_low_review = 1 if review_score <= 2 else 0

### Task 19: Select Final Columns
- Keep only 29 selected columns for the ABT

### Task 20: Inspect Final ABT
- .shape, .info(), .isnull().sum(), .duplicated().sum()

### Task 21: Create ABT Summary Report
- DataFrame with column name, dtype, missing count/%, unique values
- Save to reports/lab02_abt_summary.csv

### Task 22: Visualize Key Features
- Bar chart: Review Score Distribution -> figures/lab02_review_score_distribution.png
- Histogram: Payment Value Distribution -> figures/lab02_payment_value_distribution.png
- Bar chart: Top 10 Customer States -> figures/lab02_top_customer_states.png

### Task 23: Save Final Processed Dataset
- Save ABT to data/processed/olist_orders_abt.csv
- Verify file exists and reload to confirm

---

## 9. Expected Outputs

### data/processed/
| File | Description |
| --- | --- |
| olist_orders_abt.csv | Final Analytical Base Table (~99,000 rows x 29 columns) |

### reports/
| File | Description |
| --- | --- |
| lab02_abt_summary.csv | Column-level summary: dtype, missing %, unique values |
| lab02_abt_report.md | Markdown summary report of the ABT |

### figures/
| File | Description |
| --- | --- |
| lab02_late_delivery_distribution.png | Bar chart of on-time vs late deliveries |
| lab02_review_score_distribution.png | Bar chart of review score (1-5) frequency |
| lab02_payment_value_distribution.png | Histogram of total payment value per order |
| lab02_top_customer_states.png | Bar chart of top 10 states by order volume |

---

## 10. Reflection Questions

1. What is an Analytical Base Table and why is it important for ML?
2. Why do we aggregate payment and order item data before merging?
3. What does it mean for the ABT to be at the "order grain"?
4. Why did we define is_low_review as score <= 2 instead of score = 1?
5. What are the risks of leakage if review_score is included as a feature alongside is_low_review?
6. What percentage of orders had a late delivery? What are the business implications?
7. Which state had the most orders? What does this tell us about market concentration?
8. What is the class balance of is_low_review? Is this a problem for ML?

---

## 11. Submission Checklist

- [ ] Jupyter Notebook executed top-to-bottom without errors
- [ ] data/processed/olist_orders_abt.csv saved
- [ ] reports/lab02_abt_summary.csv saved
- [ ] reports/lab02_abt_report.md saved
- [ ] figures/lab02_late_delivery_distribution.png saved
- [ ] figures/lab02_review_score_distribution.png saved
- [ ] figures/lab02_payment_value_distribution.png saved
- [ ] figures/lab02_top_customer_states.png saved
- [ ] All reflection questions answered
