
"""
SHAP explanation functions for fraud detection.
"""
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from typing import Tuple


def compute_shap_reasons(
    model: CatBoostClassifier,
    X: pd.DataFrame
) -> Tuple[pd.DataFrame, shap.Explanation]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained CatBoostClassifier
        X: Data to compute SHAP values for
        
    Returns:
        Tuple of (shap_values_df, shap_explanation_object)
    """
    # Identify categorical features
    categorical_features = [
        "category", "merchant", "state", "gender", "city", "zip", "job"
    ]
    categorical_features = [col for col in categorical_features if col in X.columns]
    
    # Create pool
    pool = Pool(X, cat_features=categorical_features)
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pool)
    
    # Create DataFrame of SHAP values
    shap_df = pd.DataFrame(
        shap_values.values,
        columns=X.columns,
        index=X.index
    )
    
    return shap_df, shap_values

