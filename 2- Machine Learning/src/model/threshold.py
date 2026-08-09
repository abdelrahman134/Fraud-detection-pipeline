
"""
Threshold optimization functions for fraud detection.
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
from typing import List, Tuple


def find_best_threshold(
    model: CatBoostClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_folds: int = 3
) -> float:
    """
    Find the optimal classification threshold using time-series cross-validation.
    
    Args:
        model: Trained CatBoostClassifier
        X_val: Validation features
        y_val: Validation labels
        n_folds: Number of time-series folds
        
    Returns:
        Optimal threshold value
    """
    # Identify categorical features
    categorical_features = [
        "category", "merchant", "state", "gender", "city", "zip", "job"
    ]
    categorical_features = [col for col in categorical_features if col in X_val.columns]
    
    tscv = TimeSeriesSplit(n_splits=n_folds)
    thresholds = []
    
    for train_idx, val_idx in tscv.split(X_val):
        X_fold_train, X_fold_val = X_val.iloc[train_idx], X_val.iloc[val_idx]
        y_fold_train, y_fold_val = y_val.iloc[train_idx], y_val.iloc[val_idx]
        
        # Get predictions
        val_pool = Pool(X_fold_val, cat_features=categorical_features)
        y_pred_proba = model.predict_proba(val_pool)[:, 1]
        
        # Find best threshold for this fold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.1, 0.9, 81):
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_fold_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds.append(best_thresh)
    
    # Return median threshold across folds
    return float(np.median(thresholds))

