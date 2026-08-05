import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

os.makedirs('figures', exist_ok=True)

data_path = Path("data/processed/olist_orders_abt.csv")
df = pd.read_csv(data_path)

print("Part A: Load the ABT Dataset")
print("Shape:", df.shape)

print("\nPart B: Define the Prediction Problem")
target = "is_late_delivery"
print("Target column:", target)
print("Target distribution:")
print(df[target].value_counts(normalize=True) * 100)

print("\nPart C: Remove Identifier and Leakage Columns")
identifier_columns = [
    "order_id",
    "customer_id",
    "customer_unique_id"
]
leakage_columns = [
    "delivery_days",
    "delivery_delay_days",
    "review_score",
    "review_comment_count",
    "has_review_comment",
    "is_low_review",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]
columns_to_remove = identifier_columns + leakage_columns + [target]
columns_to_remove = [col for col in columns_to_remove if col in df.columns]

X = df.drop(columns=columns_to_remove)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("\nPart D: Identify Numerical and Categorical Features")
print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nPart E: Train-Test Split")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nPart L: Evaluate the Model (Logistic Regression)")
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
print("\nPart M: Inspect Transformed Feature Names")
print("Total number of transformed features:", len(feature_names))

output_model_path = Path("models/late_delivery_pipeline.joblib")
output_model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_pipeline, output_model_path)
print("\nPart N: Save the Complete Pipeline")
print("Pipeline saved at:", output_model_path)

print("\nPart U: Extension Task - Random Forest")
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))
])
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))
