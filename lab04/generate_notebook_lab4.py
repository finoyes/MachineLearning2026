import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 4: Scikit-Learn Preprocessing and Modeling Pipeline\n",
    "\n",
    "**Course**: Machine Learning\n",
    "**Topic**: Scikit-Learn Preprocessing and Modeling Pipeline\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part A: Load the ABT Dataset\n",
    "\n",
    "#### Step 1: Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
    "import joblib"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 2: Load the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "possible_paths = [\n",
    "    Path(\"../data/processed/olist_orders_abt.csv\"),\n",
    "    Path(\"data/processed/olist_orders_abt.csv\"),\n",
    "    Path(\"olist_orders_abt.csv\")\n",
    "]\n",
    "\n",
    "data_path = None\n",
    "for path in possible_paths:\n",
    "    if path.exists():\n",
    "        data_path = path\n",
    "        break\n",
    "\n",
    "if data_path is None:\n",
    "    raise FileNotFoundError(\"Could not find olist_orders_abt.csv. Please check the file path.\")\n",
    "\n",
    "df = pd.read_csv(data_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 3: Inspect Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Shape:\", df.shape)\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part B: Define the Prediction Problem\n",
    "\n",
    "#### Step 4: Check Target Column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = \"is_late_delivery\"\n",
    "\n",
    "if target not in df.columns:\n",
    "    raise ValueError(f\"Target column {target} not found in the dataset.\")\n",
    "\n",
    "df[target].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[target].value_counts(normalize=True) * 100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part C: Remove Identifier and Leakage Columns\n",
    "\n",
    "#### Step 5: Define Columns to Remove"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "identifier_columns = [\n",
    "    \"order_id\",\n",
    "    \"customer_id\",\n",
    "    \"customer_unique_id\"\n",
    "]\n",
    "\n",
    "leakage_columns = [\n",
    "    \"delivery_days\",\n",
    "    \"delivery_delay_days\",\n",
    "    \"review_score\",\n",
    "    \"review_comment_count\",\n",
    "    \"has_review_comment\",\n",
    "    \"is_low_review\",\n",
    "    \"order_delivered_customer_date\",\n",
    "    \"order_estimated_delivery_date\"\n",
    "]\n",
    "\n",
    "columns_to_remove = identifier_columns + leakage_columns + [target]\n",
    "columns_to_remove = [col for col in columns_to_remove if col in df.columns]\n",
    "\n",
    "columns_to_remove"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 6: Create Feature Matrix and Target Vector"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns=columns_to_remove)\n",
    "y = df[target]\n",
    "\n",
    "print(\"X shape:\", X.shape)\n",
    "print(\"y shape:\", y.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part D: Identify Numerical and Categorical Features\n",
    "\n",
    "#### Step 7: Identify Column Types Automatically"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_features = X.select_dtypes(include=[\"int64\", \"float64\"]).columns.tolist()\n",
    "categorical_features = X.select_dtypes(include=[\"object\", \"category\", \"bool\"]).columns.tolist()\n",
    "\n",
    "print(\"Numerical features:\")\n",
    "print(numeric_features)\n",
    "\n",
    "print(\"\\nCategorical features:\")\n",
    "print(categorical_features)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 8: Check Missing Values in Selected Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X[numeric_features].isnull().sum().sort_values(ascending=False).head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X[categorical_features].isnull().sum().sort_values(ascending=False).head(20)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part E: Train-Test Split\n",
    "\n",
    "#### Step 9: Split the Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y,\n",
    "    test_size=0.20,\n",
    "    random_state=42,\n",
    "    stratify=y\n",
    ")\n",
    "\n",
    "print(\"X_train shape:\", X_train.shape)\n",
    "print(\"X_test shape:\", X_test.shape)\n",
    "print(\"y_train shape:\", y_train.shape)\n",
    "print(\"y_test shape:\", y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part F: Build Numerical Preprocessing Pipeline\n",
    "\n",
    "#### Step 10: Create Numerical Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_pipeline = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"scaler\", StandardScaler())\n",
    "])\n",
    "\n",
    "numeric_pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part G: Build Categorical Preprocessing Pipeline\n",
    "\n",
    "#### Step 11: Create Categorical Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_pipeline = Pipeline(steps=[\n",
    "    (\"imputer\", SimpleImputer(strategy=\"constant\", fill_value=\"Unknown\")),\n",
    "    (\"encoder\", OneHotEncoder(handle_unknown=\"ignore\"))\n",
    "])\n",
    "\n",
    "categorical_pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part H: Combine Pipelines Using ColumnTransformer\n",
    "\n",
    "#### Step 12: Create Preprocessor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        (\"num\", numeric_pipeline, numeric_features),\n",
    "        (\"cat\", categorical_pipeline, categorical_features)\n",
    "    ]\n",
    ")\n",
    "\n",
    "preprocessor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part I: Create Complete Modeling Pipeline\n",
    "\n",
    "#### Step 13: Create Full Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_pipeline = Pipeline(steps=[\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"model\", LogisticRegression(max_iter=1000, class_weight=\"balanced\"))\n",
    "])\n",
    "\n",
    "model_pipeline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part J: Train the Pipeline\n",
    "\n",
    "#### Step 14: Fit the Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_pipeline.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part K: Make Predictions\n",
    "\n",
    "#### Step 15: Predict on Test Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model_pipeline.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 16: Predicted Probabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]\n",
    "y_pred_proba[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part L: Evaluate the Model\n",
    "\n",
    "#### 25. Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 26. Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "print(cm)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 27. Classification Report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part M: Inspect Transformed Feature Names\n",
    "\n",
    "#### 29. Get Feature Names After Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_names = model_pipeline.named_steps[\"preprocessor\"].get_feature_names_out()\n",
    "feature_names[:20]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(feature_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part N: Save the Complete Pipeline\n",
    "\n",
    "#### Step 17: Save the Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "output_model_path = Path(\"models/late_delivery_pipeline.joblib\")\n",
    "output_model_path.parent.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "joblib.dump(model_pipeline, output_model_path)\n",
    "print(\"Pipeline saved at:\", output_model_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 18: Load the Pipeline Again"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_pipeline = joblib.load(output_model_path)\n",
    "\n",
    "loaded_predictions = loaded_pipeline.predict(X_test)\n",
    "loaded_predictions[:10]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part O: Predict on New Data\n",
    "\n",
    "#### 32. Create a Sample New Order"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_order = X_test.iloc[[0]]\n",
    "sample_order"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_pipeline.predict(sample_order)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_pipeline.predict_proba(sample_order)[:, 1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Part U: Extension Task\n",
    "\n",
    "#### 1. Replace Logistic Regression with Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "rf_pipeline = Pipeline(steps=[\n",
    "    (\"preprocessor\", preprocessor),\n",
    "    (\"model\", RandomForestClassifier(\n",
    "        n_estimators=100,\n",
    "        random_state=42,\n",
    "        class_weight=\"balanced\"\n",
    "    ))\n",
    "])\n",
    "\n",
    "rf_pipeline.fit(X_train, y_train)\n",
    "rf_pred = rf_pipeline.predict(X_test)\n",
    "\n",
    "print(classification_report(y_test, rf_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Lab_4_Scikit_Learn_Preprocessing_and_Modeling_Pipeline.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)
