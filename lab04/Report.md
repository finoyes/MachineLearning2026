# Lab 4: Scikit-Learn Preprocessing and Modeling Pipeline

## Part R: Lab Deliverables

### Dataset Shape and Columns
**Shape:** `(99441, 29)`
The dataset contains 99,441 rows and 29 columns.

### Target Distribution Table
**Target Column:** `is_late_delivery`
| Value | Percentage |
|-------|------------|
| 0 | 96.44% |
| 1 | 3.56% |

### Removed Identifier Columns
- `order_id`
- `customer_id`
- `customer_unique_id`

### Removed Leakage Columns
- `delivery_days`
- `delivery_delay_days`
- `review_score`
- `review_comment_count`
- `has_review_comment`
- `is_low_review`
- `order_delivered_customer_date`
- `order_estimated_delivery_date`

### Numerical and Categorical Features
**Numerical Features:**
- `order_year`, `order_month`, `order_day`, `order_day_of_week`, `order_hour`, `estimated_delivery_days`, `total_payment_value`, `max_payment_installments`, `payment_types_count`, `total_items`, `total_price`, `total_freight`, `unique_products`, `unique_sellers`

**Categorical Features:**
- `customer_city`, `customer_state`, `order_status`, `dominant_payment_type`, `main_product_category`

### Pipeline Code
**Numerical Pipeline**
```python
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

**Categorical Pipeline**
```python
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

**ColumnTransformer**
```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)
```

**Complete Model Pipeline**
```python
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
```

### Logistic Regression Evaluation
**Accuracy:** `0.8592`
**Confusion Matrix:**
```text
[[16403  2777]
 [   23   686]]
```

**Classification Report:**
```text
              precision    recall  f1-score   support
           0       1.00      0.86      0.92     19180
           1       0.20      0.97      0.33       709
    accuracy                           0.86     19889
   macro avg       0.60      0.91      0.63     19889
weighted avg       0.97      0.86      0.90     19889
```

### Explanation of Pipeline Leakage Prevention
A pipeline prevents data leakage by ensuring that all data transformations (like imputing missing values or scaling features) learn their parameters (e.g., mean, median, standard deviation) strictly from the training dataset. When transforming the test set or new data, it consistently applies those previously learned parameters rather than calculating new ones, ensuring the test set doesn't inappropriately influence the model's training process.

### Saved Pipeline
Pipeline saved at: `models/late_delivery_pipeline.joblib`

## Part S: Student Reflection Questions
1. **What is the main purpose of a Scikit-Learn pipeline?**
   To organize sequential preprocessing and modeling steps into a single reusable object, reducing code redundancy, preventing data leakage, and ensuring reproducibility.
2. **Why should preprocessing be fitted only on training data?**
   To prevent data leakage. If test data is used to fit preprocessing transformations (like scaling or imputing), information from the test set leaks into the training process, leading to overly optimistic evaluations.
3. **What is the role of ColumnTransformer?**
   It allows different preprocessing pipelines to be applied to different subsets of features (e.g., handling numerical and categorical columns distinctly) before combining them into a single feature matrix.
4. **Why do numerical and categorical columns need different preprocessing?**
   Numerical columns require mathematical transformations like scaling and mean/median imputation, while categorical text/labels cannot be mathematically scaled and require transformations like one-hot encoding or constant imputation.
5. **Why should identifiers not be used as features?**
   They don't contain generalized predictive patterns and can cause the model to simply memorize specific training instances instead of learning underlying relationships.
6. **Why is delivery_delay_days a leakage column?**
   Because this information is not known until the order has actually been delivered. Using it as a feature would directly reveal the target outcome before the prediction event naturally occurs.
7. **Why is OneHotEncoder(handle_unknown="ignore") useful?**
   It prevents the model from throwing an error in production if it encounters a new, previously unseen categorical value that was not present in the training data.
8. **Why did we use median imputation for numerical columns?**
   Median imputation is more robust to outliers compared to mean imputation, which is particularly useful for skewed real-world business data like sales and logistics metrics.
9. **Which algorithms are sensitive to scaling?**
   Algorithms that rely on distance metrics or gradient descent, such as Logistic Regression, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Neural Networks.
10. **Why should we save the complete pipeline and not only the model?**
    Saving only the model neglects the specific preprocessing logic it expects. Saving the entire pipeline guarantees that any incoming raw data in the future is preprocessed in exactly the same way before being passed to the trained estimator.

## Part U: Extension Task - Random Forest

**Random Forest Classification Report:**
```text
              precision    recall  f1-score   support
           0       0.96      1.00      0.98     19180
           1       0.00      0.00      0.00       709
    accuracy                           0.96     19889
   macro avg       0.48      0.50      0.49     19889
weighted avg       0.93      0.96      0.95     19889
```

### Discussion
1. **Which model performed better?**
   While Random Forest achieved a higher raw accuracy (0.96 vs 0.86), it completely failed to identify the minority class (Recall = 0.00 for class 1). Logistic Regression, with `class_weight="balanced"`, successfully achieved a high recall (0.97) for the late delivery class, making it much better suited for solving the actual business problem.
2. **Which model is easier to interpret?**
   Logistic Regression is much easier to interpret, as it produces linear coefficients that denote the individual impact of each feature on the final log-odds prediction.
3. **Does Random Forest require scaling?**
   No, tree-based models make binary splits on feature values, so they are immune to differing scales or magnitudes across columns.
4. **Why did we still keep preprocessing in the pipeline?**
   Even though scaling isn't necessary for Random Forests, we still need the pipeline for proper missing value imputation, one-hot encoding of categorical variables, and ensuring a robust, organized workflow for new incoming data.
