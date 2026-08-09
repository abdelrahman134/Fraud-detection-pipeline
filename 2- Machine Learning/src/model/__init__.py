
"""
Model training and evaluation module for fraud detection.
"""
from .train import train_model
from .evaluate import evaluate_model
from .threshold import find_best_threshold
from .shap_explainer import compute_shap_reasons

__all__ = [
    "train_model",
    "evaluate_model",
    "find_best_threshold",
    "compute_shap_reasons"
]

