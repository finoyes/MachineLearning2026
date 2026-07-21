import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Cell 1
cells.append(nbf.v4.new_markdown_cell("# Part A: Load and Inspect the Dataset"))

# Cell 2
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)"""))

# Cell 3
cells.append(nbf.v4.new_code_cell("""# If your notebook is inside the notebooks/ folder, use:
df = pd.read_csv("../data/processed/olist_orders_abt.csv")
df.head()"""))

# Cell 4
cells.append(nbf.v4.new_code_cell("""df.shape"""))

# Cell 5
cells.append(nbf.v4.new_code_cell("""df.info()"""))

# Cell 6
cells.append(nbf.v4.new_code_cell("""df.describe()"""))

# Cell 7
cells.append(nbf.v4.new_code_cell("""# Student Task 1: Create Data Type Lists
nominal_columns = [
    "customer_state",
    "customer_city",
    "main_product_category",
    "dominant_payment_type",
    "order_status"
]

ordinal_columns = [
    "review_score"
]

interval_columns = [
    "order_hour",
    "order_month",
    "order_day_of_week"
]

ratio_columns = [
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "estimated_delivery_days",
    "delivery_delay_days",
    "total_items",
    "unique_products",
    "unique_sellers"
]"""))

# Cell 8
cells.append(nbf.v4.new_code_cell("""df.columns"""))

# Cell 9
cells.append(nbf.v4.new_code_cell("""def keep_existing_columns(dataframe, columns):
    return [col for col in columns if col in dataframe.columns]

nominal_columns = keep_existing_columns(df, nominal_columns)
ordinal_columns = keep_existing_columns(df, ordinal_columns)
interval_columns = keep_existing_columns(df, interval_columns)
ratio_columns = keep_existing_columns(df, ratio_columns)"""))

# Cell 10
cells.append(nbf.v4.new_markdown_cell("# Part C: Create Additional Teaching-Friendly Features"))

# Cell 11
cells.append(nbf.v4.new_code_cell("""if "delivery_days" in df.columns:
    df["delivery_speed_category"] = pd.cut(
        df["delivery_days"],
        bins=[-1, 7, 15, 30, np.inf],
        labels=["Fast", "Normal", "Slow", "Very Delayed"]
    )"""))

# Cell 12
cells.append(nbf.v4.new_code_cell("""if "total_payment_value" in df.columns:
    df["payment_value_band"] = pd.qcut(
        df["total_payment_value"],
        q=4,
        labels=["Low", "Medium", "High", "Premium"],
        duplicates="drop"
    )"""))

# Cell 13
cells.append(nbf.v4.new_markdown_cell("# Part D: Missing Value Analysis"))

# Cell 14
cells.append(nbf.v4.new_code_cell("""df.isnull().sum()"""))

# Cell 15
cells.append(nbf.v4.new_code_cell("""missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().sum() / len(df)) * 100
}).sort_values(by="missing_percent", ascending=False)

missing_summary.head(20)"""))

# Cell 16
cells.append(nbf.v4.new_code_cell("""missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

sns.barplot(
    x=missing_values.values,
    y=missing_values.index
)

plt.title("Missing Values by Column")
plt.xlabel("Number of Missing Values")
plt.ylabel("Column")
plt.show()"""))

# Cell 17
cells.append(nbf.v4.new_code_cell("""sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.show()"""))

# Cell 18
cells.append(nbf.v4.new_markdown_cell("# Part E: Univariate Analysis"))

# Cell 19
cells.append(nbf.v4.new_code_cell("""if "customer_state" in df.columns:
    sns.countplot(
        data=df,
        x="customer_state",
        order=df["customer_state"].value_counts().index
    )
    plt.title("Orders by Customer State")
    plt.xlabel("Customer State")
    plt.ylabel("Number of Orders")
    plt.xticks(rotation=45)
    plt.show()"""))

# Cell 20
cells.append(nbf.v4.new_code_cell("""if "main_product_category" in df.columns:
    top_categories = df["main_product_category"].value_counts().head(15).index
    
    sns.countplot(
        data=df[df["main_product_category"].isin(top_categories)],
        y="main_product_category",
        order=top_categories
    )
    
    plt.title("Top Product Categories")
    plt.xlabel("Number of Orders")
    plt.ylabel("Product Category")
    plt.show()"""))

# Cell 21
cells.append(nbf.v4.new_code_cell("""if "delivery_speed_category" in df.columns:
    delivery_order = ["Fast", "Normal", "Slow", "Very Delayed"]
    
    sns.countplot(
        data=df,
        x="delivery_speed_category",
        order=delivery_order
    )
    
    plt.title("Delivery Speed Category Distribution")
    plt.xlabel("Delivery Speed Category")
    plt.ylabel("Number of Orders")
    plt.show()"""))

# Cell 22
cells.append(nbf.v4.new_code_cell("""if "total_payment_value" in df.columns:
    sns.histplot(
        data=df,
        x="total_payment_value",
        bins=50,
        kde=True
    )
    
    plt.title("Distribution of Total Payment Value")
    plt.xlabel("Total Payment Value")
    plt.ylabel("Frequency")
    plt.show()"""))

# Cell 23
cells.append(nbf.v4.new_code_cell("""if "delivery_days" in df.columns:
    sns.histplot(
        data=df,
        x="delivery_days",
        bins=40,
        kde=True
    )
    
    plt.title("Distribution of Delivery Days")
    plt.xlabel("Delivery Days")
    plt.ylabel("Frequency")
    plt.show()"""))

# Cell 24
cells.append(nbf.v4.new_markdown_cell("# Part F: Skewness"))

# Cell 25
cells.append(nbf.v4.new_code_cell("""skew_cols = [
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "delivery_delay_days",
    "total_items"
]"""))

# Cell 26
cells.append(nbf.v4.new_code_cell("""skew_cols = keep_existing_columns(df, skew_cols)

skewness_values = df[skew_cols].skew().sort_values(ascending=False)

skewness_values"""))

# Cell 27
cells.append(nbf.v4.new_code_cell("""sns.barplot(
    x=skewness_values.values,
    y=skewness_values.index
)

plt.title("Skewness of Numerical Features")
plt.xlabel("Skewness")
plt.ylabel("Feature")
plt.show()"""))

# Cell 28
cells.append(nbf.v4.new_markdown_cell("# Part G: Box Plot and Outlier Detection"))

# Cell 29
cells.append(nbf.v4.new_code_cell("""if "total_payment_value" in df.columns:
    sns.boxplot(
        data=df,
        y="total_payment_value"
    )
    
    plt.title("Box Plot of Total Payment Value")
    plt.ylabel("Total Payment Value")
    plt.show()"""))

# Cell 30
cells.append(nbf.v4.new_code_cell("""if "delivery_days" in df.columns:
    sns.boxplot(
        data=df,
        y="delivery_days"
    )
    
    plt.title("Box Plot of Delivery Days")
    plt.ylabel("Delivery Days")
    plt.show()"""))

# Cell 31
cells.append(nbf.v4.new_code_cell("""def iqr_outlier_summary(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = dataframe[
        (dataframe[column] < lower_bound) |
        (dataframe[column] > upper_bound)
    ]
    
    summary = {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_percent": (len(outliers) / len(dataframe)) * 100
    }
    
    return summary"""))

# Cell 32
cells.append(nbf.v4.new_code_cell("""if "total_payment_value" in df.columns:
    print(iqr_outlier_summary(df, "total_payment_value"))"""))

# Cell 33
cells.append(nbf.v4.new_code_cell("""if "delivery_days" in df.columns:
    print(iqr_outlier_summary(df, "delivery_days"))"""))

# Cell 34
cells.append(nbf.v4.new_markdown_cell("# Part H: Multivariate Analysis"))

# Cell 35
cells.append(nbf.v4.new_code_cell("""if {"delivery_delay_days", "review_score"}.issubset(df.columns):
    sns.scatterplot(
        data=df,
        x="delivery_delay_days",
        y="review_score",
        alpha=0.6
    )
    
    plt.title("Delivery Delay Days vs Review Score")
    plt.xlabel("Delivery Delay Days")
    plt.ylabel("Review Score")
    plt.show()"""))

# Cell 36
cells.append(nbf.v4.new_code_cell("""if {"total_payment_value", "total_freight", "is_late_delivery"}.issubset(df.columns):
    sns.scatterplot(
        data=df,
        x="total_payment_value",
        y="total_freight",
        hue="is_late_delivery",
        alpha=0.7
    )
    
    plt.title("Payment Value vs Freight by Late Delivery")
    plt.xlabel("Total Payment Value")
    plt.ylabel("Total Freight")
    plt.show()"""))

# Cell 37
cells.append(nbf.v4.new_code_cell("""if {"is_late_delivery", "review_score"}.issubset(df.columns):
    sns.boxplot(
        data=df,
        x="is_late_delivery",
        y="review_score"
    )
    
    plt.title("Review Score by Late Delivery Status")
    plt.xlabel("Is Late Delivery")
    plt.ylabel("Review Score")
    plt.show()"""))

# Cell 38
cells.append(nbf.v4.new_code_cell("""numeric_columns = [
    "order_hour",
    "order_month",
    "order_day_of_week",
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "estimated_delivery_days",
    "delivery_delay_days",
    "total_items",
    "unique_products",
    "unique_sellers",
    "review_score",
    "review_comment_count",
    "has_review_comment",
    "is_late_delivery",
    "is_low_review"
]

numeric_columns = keep_existing_columns(df, numeric_columns)

corr = df[numeric_columns].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()"""))

# Cell 39
cells.append(nbf.v4.new_markdown_cell("# Part I: Target Variable Analysis"))

# Cell 40
cells.append(nbf.v4.new_code_cell("""if "is_late_delivery" in df.columns:
    sns.countplot(
        data=df,
        x="is_late_delivery"
    )
    
    plt.title("Target Distribution: Late Delivery")
    plt.xlabel("Is Late Delivery")
    plt.ylabel("Number of Orders")
    plt.show()"""))

# Cell 41
cells.append(nbf.v4.new_code_cell("""if "is_low_review" in df.columns:
    sns.countplot(
        data=df,
        x="is_low_review"
    )
    
    plt.title("Target Distribution: Low Review")
    plt.xlabel("Is Low Review")
    plt.ylabel("Number of Orders")
    plt.show()"""))

# Cell 42
cells.append(nbf.v4.new_markdown_cell("# Part J: Fixing Data Problems"))

# Cell 43
cells.append(nbf.v4.new_code_cell("""clean_df = df.copy()"""))

# Cell 44
cells.append(nbf.v4.new_code_cell("""if "delivery_days" in clean_df.columns:
    clean_df["missing_delivery_days"] = clean_df["delivery_days"].isnull().astype(int)"""))

# Cell 45
cells.append(nbf.v4.new_code_cell("""categorical_missing_cols = [
    "main_product_category",
    "dominant_payment_type",
    "delivery_speed_category",
    "payment_value_band",
    "customer_state",
    "customer_city",
    "order_status"
]

categorical_missing_cols = keep_existing_columns(clean_df, categorical_missing_cols)

for col in categorical_missing_cols:
    if clean_df[col].dtype.name == 'category':
        if "Unknown" not in clean_df[col].cat.categories:
            clean_df[col] = clean_df[col].cat.add_categories("Unknown")
    clean_df[col] = clean_df[col].fillna("Unknown")"""))

# Cell 46
cells.append(nbf.v4.new_code_cell("""numerical_missing_cols = [
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "estimated_delivery_days",
    "delivery_delay_days",
    "review_score",
    "total_items",
    "unique_products",
    "unique_sellers"
]

numerical_missing_cols = keep_existing_columns(clean_df, numerical_missing_cols)

for col in numerical_missing_cols:
    clean_df[col] = clean_df[col].fillna(clean_df[col].median())"""))

# Cell 47
cells.append(nbf.v4.new_code_cell("""if "total_payment_value" in clean_df.columns:
    clean_df["log_total_payment_value"] = np.log1p(clean_df["total_payment_value"])

if "total_freight" in clean_df.columns:
    clean_df["log_total_freight"] = np.log1p(clean_df["total_freight"])

if "total_price" in clean_df.columns:
    clean_df["log_total_price"] = np.log1p(clean_df["total_price"])"""))

# Cell 48
cells.append(nbf.v4.new_code_cell("""if {"total_payment_value", "log_total_payment_value"}.issubset(clean_df.columns):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(
        data=clean_df,
        x="total_payment_value",
        bins=50,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_title("Original Total Payment Value")
    
    sns.histplot(
        data=clean_df,
        x="log_total_payment_value",
        bins=50,
        kde=True,
        ax=axes[1]
    )
    axes[1].set_title("Log-Transformed Total Payment Value")
    
    plt.show()"""))

# Cell 49
cells.append(nbf.v4.new_markdown_cell("# Part K: Scaling Demonstration"))

# Cell 50
cells.append(nbf.v4.new_code_cell("""scale_check_cols = [
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "total_items"
]

scale_check_cols = keep_existing_columns(clean_df, scale_check_cols)

clean_df[scale_check_cols].describe()"""))

# Cell 51
cells.append(nbf.v4.new_code_cell("""from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaling_cols = [
    "total_payment_value",
    "total_price",
    "total_freight",
    "delivery_days",
    "total_items"
]

scaling_cols = keep_existing_columns(clean_df, scaling_cols)

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

standard_scaled = standard_scaler.fit_transform(clean_df[scaling_cols])
minmax_scaled = minmax_scaler.fit_transform(clean_df[scaling_cols])
robust_scaled = robust_scaler.fit_transform(clean_df[scaling_cols])

standard_scaled_df = pd.DataFrame(
    standard_scaled,
    columns=[col + "_standard_scaled" for col in scaling_cols]
)

minmax_scaled_df = pd.DataFrame(
    minmax_scaled,
    columns=[col + "_minmax_scaled" for col in scaling_cols]
)

robust_scaled_df = pd.DataFrame(
    robust_scaled,
    columns=[col + "_robust_scaled" for col in scaling_cols]
)

standard_scaled_df.head()"""))

# Cell 52
cells.append(nbf.v4.new_markdown_cell("""# Part L: Data Leakage Discussion

Leakage Example
Suppose our target is:
`is_late_delivery`

This target is created from:
`delivery_delay_days`

If we use `delivery_delay_days` to predict `is_late_delivery`, the model is cheating."""))

# Cell 53
cells.append(nbf.v4.new_code_cell("""leakage_columns = [
    "delivery_delay_days",
    "delivery_speed_category"
]"""))

# Cell 54
cells.append(nbf.v4.new_markdown_cell("# Part M: Save Cleaned ABT"))

# Cell 55
cells.append(nbf.v4.new_code_cell("""clean_df.to_csv("../data/processed/olist_orders_abt_cleaned.csv", index=False)

print("Cleaned ABT saved successfully.")
print(clean_df.shape)"""))

nb['cells'] = cells

with open('notebooks/Lab03_Data_Quality_EDA_Preprocessing.ipynb', 'w') as f:
    nbf.write(nb, f)
