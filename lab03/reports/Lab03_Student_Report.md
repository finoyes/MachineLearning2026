# Student Report

**Name:** [Student Name]
**Roll Number:** [Roll Number]
**Batch:** [Batch]
**Date:** [Date]
**Lab Number:** 3
**Lab Title:** Data Quality Assessment, EDA, and Preprocessing

## 1. Dataset Description
The dataset used is the Analytical Base Table (`olist_orders_abt.csv`) generated from the Olist e-commerce relational tables. Each row in the dataset represents one customer order. The goal is to perform data quality assessment, exploratory data analysis, and preprocessing before applying machine learning algorithms.

## 2. Data Type Identification
Based on the columns available:
- **Nominal Data:** `customer_state`, `customer_city`, `main_product_category`, `dominant_payment_type`, `order_status`
- **Ordinal Data:** `review_score`, `delivery_speed_category`, `payment_value_band`
- **Interval Data:** `order_hour`, `order_month`, `order_day_of_week`
- **Ratio Data:** `total_payment_value`, `total_price`, `total_freight`, `delivery_days`, `estimated_delivery_days`, `delivery_delay_days`, `total_items`, `unique_products`, `unique_sellers`

## 3. Missing Value Analysis
During the analysis, some columns were found to have missing values (e.g., `delivery_days`, categorical columns like `main_product_category`, etc.). 
- A missing value summary table and heatmap were generated to visualize the missingness. 
- **Data Issue 1:** Delivery information is missing for some orders. We added a missingness indicator column `missing_delivery_days` instead of simply dropping them to preserve information.

## 4. Univariate Analysis
- **Count Plots:** Showed the distribution of categorical variables such as `customer_state` and top product categories.
- **Histograms:** Displayed the distribution of numerical variables like `total_payment_value` and `delivery_days`. 
- **Data Issue 2:** The distribution of numerical features like `total_payment_value` is heavily right-skewed, meaning most orders are small, but there are a few very large extreme values pulling the tail to the right.

## 5. Multivariate Analysis
- **Scatter Plots:** Explored relationships between `delivery_delay_days` and `review_score`, as well as `total_payment_value` vs `total_freight`.
- **Correlation Heatmap:** Visualized correlations between all numerical features to see how variables move together.
- We observed potential relationships between delivery delays and customer review scores.

## 6. Outlier and Skewness Analysis
- Box plots and IQR methods detected significant outliers in `total_payment_value` and `delivery_days`.
- **Data Issue 3:** Large extreme values (outliers) heavily impact the mean and standard scaling of numerical features, making the data less reliable for certain algorithms without proper transformation.

## 7. Data Cleaning Strategy
| Data Issue | Suggested Fixing Strategy |
| --- | --- |
| Missing Categorical Values | Imputed with `"Unknown"` to preserve the missingness meaning. |
| Missing Numerical Values | Imputed using the `median` because the distributions are skewed (mean would be affected by outliers). |
| Skewness in Payment/Price | Applied Log1p Transformation (`np.log1p`) to normalize the skewed distributions. |
| Missing Delivery Days | Created a missing indicator `missing_delivery_days` before imputation. |

## 8. Data Leakage Discussion
**Which column may cause data leakage and why?**
If we are predicting `is_late_delivery` (whether an order is late or not), columns like `delivery_delay_days` and `delivery_speed_category` directly reveal information about the actual delivery timeline. Since these features are not known at the time of prediction (before the order is delivered), using them will cause data leakage. The model would "cheat" by using future information, resulting in artificially high accuracy during training but failure in the real world.

## 9. Conclusion
In this lab, we successfully loaded the ABT, identified various data types, handled missing values, detected outliers and skewness, and applied necessary transformations (log transformation and robust scaling). The dataset is now cleaned and preprocessed (`olist_orders_abt_cleaned.csv`), making it suitable for training reliable Machine Learning models.
