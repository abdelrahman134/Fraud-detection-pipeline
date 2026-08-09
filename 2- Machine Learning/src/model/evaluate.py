
"""
Model evaluation functions for fraud detection.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, Tuple


def evaluate_model(
    model: CatBoostClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained CatBoostClassifier
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Identify categorical features
    categorical_features = [
        "category", "merchant", "state", "gender", "city", "zip", "job"
    ]
    categorical_features = [col for col in categorical_features if col in X_test.columns]
    
    # Create test pool
    test_pool = Pool(X_test, cat_features=categorical_features)
    
    # Get predictions
    y_pred_proba = model.predict_proba(test_pool)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        "auc": roc_auc_score(y_test, y_pred_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics

