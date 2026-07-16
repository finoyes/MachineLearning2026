# Lab 2 – Analytical Base Table Report

**Prepared by:** Sharad Laad  
**LinkedIn:** https://www.linkedin.com/in/sharadlaad/  
**Dataset:** Olist Brazilian E-Commerce Public Dataset  
**Lab:** Lab 2 – Building an Analytical Base Table  

---

## 1. ABT Overview

| Metric | Value |
| --- | --- |
| Total Orders (rows) | 99,441 |
| Total Features (columns) | 29 |
| Target Variable | is_low_review |
| Target Type | Binary Classification |

---

## 2. Column Summary

| Column | Data Type | Missing # | Missing % | Unique Values |
| --- | --- | --- | --- | --- |
| order_id | object | 0 | 0.0% | 99,441 |
| customer_id | object | 0 | 0.0% | 62,727 |
| customer_unique_id | object | 0 | 0.0% | 62,727 |
| customer_city | object | 0 | 0.0% | 18 |
| customer_state | object | 0 | 0.0% | 20 |
| order_status | object | 0 | 0.0% | 8 |
| order_year | int32 | 0 | 0.0% | 3 |
| order_month | int32 | 0 | 0.0% | 12 |
| order_day | int32 | 0 | 0.0% | 31 |
| order_day_of_week | int32 | 0 | 0.0% | 7 |
| order_hour | int32 | 0 | 0.0% | 24 |
| delivery_days | float64 | 3,334 | 3.4% | 23 |
| estimated_delivery_days | int64 | 0 | 0.0% | 31 |
| delivery_delay_days | float64 | 3,334 | 3.4% | 53 |
| is_late_delivery | int64 | 0 | 0.0% | 2 |
| review_score | float64 | 19,853 | 20.0% | 5 |
| is_low_review | int64 | 0 | 0.0% | 2 |
| review_comment_count | float64 | 19,853 | 20.0% | 2 |
| has_review_comment | float64 | 19,853 | 20.0% | 2 |
| total_payment_value | float64 | 0 | 0.0% | 37,071 |
| max_payment_installments | int64 | 0 | 0.0% | 6 |
| payment_types_count | int64 | 0 | 0.0% | 3 |
| dominant_payment_type | object | 0 | 0.0% | 4 |
| total_items | int64 | 0 | 0.0% | 5 |
| total_price | float64 | 0 | 0.0% | 32,921 |
| total_freight | float64 | 0 | 0.0% | 15,865 |
| unique_products | int64 | 0 | 0.0% | 5 |
| unique_sellers | int64 | 0 | 0.0% | 5 |
| main_product_category | object | 2,085 | 2.1% | 20 |

---

## 3. Key Findings

### 3.1 Delivery Performance

| Metric | Value |
| --- | --- |
| Average Delivery Days | 11.7 days |
| Late Deliveries | 3,543 orders (3.6%) |
| On-Time Deliveries | 95,898 orders (96.4%) |

### 3.2 Target Variable Distribution

| Class | Count | Percentage |
| --- | --- | --- |
| is_low_review = 1 (Low: score ≤ 2) | 11,162 | 11.2% |
| is_low_review = 0 (High: score > 2) | 88,279 | 88.8% |

> **Note:** Orders without a review score (20.0% of total) are labelled 0 by default
> because `np.where` treats NaN as the "else" branch.

### 3.3 Customer Geography

| Metric | Value |
| --- | --- |
| Top State | SP (41,281 orders) |
| Unique States | 20 |
| Unique Cities | 18 |

### 3.4 Payments

| Metric | Value |
| --- | --- |
| Average Order Value (BRL) | R$ 153.98 |
| Max Order Value (BRL) | R$ 4289.60 |
| Unique Payment Types | 4 |

### 3.5 Products

| Metric | Value |
| --- | --- |
| Top Product Category | auto |
| Unique Categories | 20 |
| Avg Items per Order | 1.49 |

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
Approximately 3.6% of delivered orders arrived after the estimated delivery date. This has significant business implications: late deliveries are a leading predictor of low reviews and customer churn.

**Q7. Which state had the most orders?**  
SP dominated with 41,281 orders, revealing high geographic concentration in Brazil's southeast. This means ML models may be biased toward patterns from this region.

**Q8. What is the class balance of is_low_review?**  
Approximately 11.2% of orders have is_low_review = 1. This moderate class imbalance should be addressed in future labs using techniques like class weighting, oversampling (SMOTE), or threshold tuning during model evaluation.

---

## 6. Output Files

```
data/processed/olist_orders_abt.csv   — Final ABT (99,441 rows × 29 cols)
reports/lab02_abt_summary.csv         — Column-level metadata
reports/lab02_abt_report.md           — This report
figures/lab02_late_delivery_distribution.png
figures/lab02_review_score_distribution.png
figures/lab02_payment_value_distribution.png
figures/lab02_top_customer_states.png
```
