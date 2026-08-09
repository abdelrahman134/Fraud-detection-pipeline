
# Fraud Detection Pipeline Guide

## Overview

This pipeline detects fraudulent transactions using machine learning on Databricks. The pipeline consists of three main notebooks:
1. **EDA** - Exploratory data analysis
2. **Feature Engineering** - Building predictive features
3. **Model Training** - Training and evaluating the fraud detection model

## Project Structure

```
Databricks/
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering and data processing
│   └── 03_model_training.ipynb       # Model training, evaluation, and MLflow logging
│
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── features/
│   │   ├── static_features.py        # Basic feature transformations
│   │   ├── window_features.py        # Time-based window features
│   │   ├── geospatial_features.py    # Distance and location features
│   │   └── lookup_features.py        # Lookup table-based features
│   └── model/
│       ├── train.py                  # Model training functions
│       ├── evaluate.py               # Model evaluation metrics
│       ├── threshold.py              # Optimal threshold selection
│       └── shap_explainer.py         # SHAP value explanations
│
├── tests/
│   ├── test_static_features.py
│   ├── test_window_features.py
│   └── test_evaluate.py
│
├── config/
│   └── pipeline_config.yaml          # Pipeline configuration
│
└── PIPELINE_GUIDE.md                 # This file
```

## How to Run on Databricks

### 1. Set Up the Environment

1. Create a Databricks cluster with the following libraries installed:
   - `catboost`
   - `mlflow`
   - `shap`
   - `pyspark` (already included in Databricks Runtime)

2. Upload the `Databricks/` directory to your Databricks workspace.

### 2. Run Notebooks in Order

1. **01_eda.ipynb**: Explores the raw data and visualizes key patterns
2. **02_feature_engineering.ipynb**: Builds features and saves processed data
3. **03_model_training.ipynb**: Trains the model, evaluates it, and logs to MLflow

### 3. Configure as a Databricks Job

To run the pipeline automatically:

1. Create a new Databricks Job
2. Add three tasks in order:
   - Task 1: `01_eda.ipynb`
   - Task 2: `02_feature_engineering.ipynb` (depends on Task 1)
   - Task 3: `03_model_training.ipynb` (depends on Task 2)
3. Configure job parameters for S3 paths:
   - `raw_data_path`: S3 path to raw parquet files
   - `processed_data_path`: S3 path to save processed data

## Input/Output Data

### Input Data
- **Raw Data**: Parquet files containing transaction records with fields like `cc_num`, `trans_date`, `amt`, `is_fraud`, etc.

### Output Data
- **Processed Data**: Parquet files with all engineered features
- **MLflow Model**: Trained CatBoost model logged to MLflow with metrics and parameters

## Features

The pipeline creates the following feature groups:

1. **Static Features**:
   - Time-based (hour, day of week, month)
   - Transaction amount
   - Geographic distance

2. **Window Features**:
   - Transaction count in last 1/24 hours
   - Average spend in last 24 hours
   - Unique merchants in last 24 hours
   - Time since last transaction

3. **Geospatial Features**:
   - Haversine distance between customer and merchant

4. **Lookup Features**:
   - Fraud rate by category/merchant/state (from training data)

## Model

The model uses **CatBoostClassifier** with:
- Time-series cross-validation for threshold selection
- Early stopping to prevent overfitting
- Categorical feature support built-in
- MLflow for experiment tracking

## Configuration

All configuration is in `config/pipeline_config.yaml`:
- Model parameters (iterations, learning rate, depth)
- Data paths (raw, processed, test)
- Threshold optimization settings
- MLflow configuration

