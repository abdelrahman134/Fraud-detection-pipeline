
"""
Test model evaluation.
"""
import pytest
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import sys
import os

sys.path.append(os.path.abspath('../src'))
from model.evaluate import evaluate_model


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "amt": np.random.uniform(1, 1000, 100),
        "city_pop": np.random.randint(1000, 1000000, 100),
        "hour": np.random.randint(0, 24, 100),
    })
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def trained_model(sample_data):
    X, y = sample_data
    model = CatBoostClassifier(iterations=10, verbose=False, random_state=42)
    model.fit(X, y)
    return model


def test_evaluate_model(trained_model, sample_data):
    X, y = sample_data
    metrics = evaluate_model(trained_model, X, y, threshold=0.5)
    
    # Check all metrics are present
    assert "auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "threshold" in metrics
    assert "confusion_matrix" in metrics
    
    # Check metric values are reasonable
    assert 0 <= metrics["auc"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert len(metrics["confusion_matrix"]) == 2

