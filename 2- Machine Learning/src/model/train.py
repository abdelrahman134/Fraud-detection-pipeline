
"""
Model training functions for fraud detection.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict, Any


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: Dict[str, Any] = None
) -> CatBoostClassifier:
    """
    Train a CatBoost classifier for fraud detection.
    
    Args:
        X_train: Training features (pandas DataFrame)
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_params: Dictionary of CatBoost parameters
        
    Returns:
        Trained CatBoostClassifier
    """
    if model_params is None:
        model_params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
            "verbose": 100,
            "random_state": 42
        }
    
    # Identify categorical features
    categorical_features = [
        "category", "merchant", "state", "gender", "city", "zip", "job"
    ]
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    
    # Create CatBoost pools
    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    val_pool = Pool(X_val, y_val, cat_features=categorical_features)
    
    # Train the model
    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=val_pool)
    
    return model

