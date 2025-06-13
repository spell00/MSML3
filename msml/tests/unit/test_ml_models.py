import pytest
import numpy as np
from msml.ml.models import (
    Train_xgboost,
    Train_bernn,
    TrainAEClassifierHoldout
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_xgboost_train(sample_data):
    """Test XGBoost training"""
    X, y = sample_data
    model = Train_xgboost()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_bernn_train(sample_data):
    """Test BERNN training"""
    X, y = sample_data
    model = Train_bernn()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_ae_classifier_holdout(sample_data):
    """Test AE Classifier with holdout"""
    X, y = sample_data
    model = TrainAEClassifierHoldout()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_model_save_load(sample_data, tmp_path):
    """Test model save and load functionality"""
    X, y = sample_data
    model = Train_xgboost()
    model.fit(X, y)

    # Save model
    save_path = tmp_path / "model.pkl"
    model.save(save_path)

    # Load model
    loaded_model = Train_xgboost()
    loaded_model.load(save_path)

    # Compare predictions
    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(original_preds, loaded_preds)
