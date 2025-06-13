import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from msml.ml.train_xgboost import Train_xgboost


@pytest.fixture
def sample_xgboost_data():
    """Create sample data for XGBoost training"""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    data = {
        'inputs': {
            'all': pd.DataFrame(np.random.randn(n_samples, n_features)),
            'train': pd.DataFrame(np.random.randn(60, n_features)),
            'valid': pd.DataFrame(np.random.randn(20, n_features)),
            'test': pd.DataFrame(np.random.randn(20, n_features))
        },
        'labels': {
            'all': np.array(['class1'] * 50 + ['class2'] * 50),
            'train': np.array(['class1'] * 30 + ['class2'] * 30),
            'valid': np.array(['class1'] * 10 + ['class2'] * 10),
            'test': np.array(['class1'] * 10 + ['class2'] * 10)
        },
        'batches': {
            'all': np.array(['batch1'] * 50 + ['batch2'] * 50),
            'train': np.array(['batch1'] * 30 + ['batch2'] * 30),
            'valid': np.array(['batch1'] * 10 + ['batch2'] * 10),
            'test': np.array(['batch1'] * 10 + ['batch2'] * 10)
        }
    }
    return data


def test_xgboost_initialization(sample_xgboost_data):
    """Test XGBoost training class initialization"""
    train = Train_xgboost(
        name='test_xgboost',
        model=xgb.XGBClassifier(),
        data=sample_xgboost_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False
    )
    assert train.name == 'test_xgboost'
    assert isinstance(train.model, xgb.XGBClassifier)


def test_get_xgboost_model(sample_xgboost_data):
    """Test XGBoost model creation"""
    train = Train_xgboost(
        name='test_xgboost',
        model=xgb.XGBClassifier(),
        data=sample_xgboost_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False
    )

    dmatrices = {
        'train': xgb.DMatrix(sample_xgboost_data['inputs']['train'],
                             label=sample_xgboost_data['labels']['train']),
        'valid': xgb.DMatrix(sample_xgboost_data['inputs']['valid'],
                             label=sample_xgboost_data['labels']['valid'])
    }

    param_grid = {
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100
    }

    model = train.get_xgboost_model(dmatrices, param_grid)
    assert isinstance(model, xgb.XGBClassifier)


def test_xgboost_training(sample_xgboost_data):
    """Test XGBoost training process"""
    train = Train_xgboost(
        name='test_xgboost',
        model=xgb.XGBClassifier(),
        data=sample_xgboost_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False
    )

    h_params = {
        'max_depth': [3],
        'learning_rate': [0.1],
        'n_estimators': [100]
    }

    train.train(h_params)
    assert hasattr(train, 'best_model')


def test_xgboost_prediction(sample_xgboost_data):
    """Test XGBoost prediction functionality"""
    train = Train_xgboost(
        name='test_xgboost',
        model=xgb.XGBClassifier(),
        data=sample_xgboost_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False
    )

    # Train the model first
    h_params = {
        'max_depth': [3],
        'learning_rate': [0.1],
        'n_estimators': [100]
    }
    train.train(h_params)

    # Make predictions
    predictions = train.make_predictions(sample_xgboost_data, None)
    assert len(predictions) > 0


def test_xgboost_save_load(sample_xgboost_data, tmp_path):
    """Test XGBoost model saving and loading"""
    train = Train_xgboost(
        name='test_xgboost',
        model=xgb.XGBClassifier(),
        data=sample_xgboost_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path=str(tmp_path),
        args=None,
        logger=None,
        log_neptune=False
    )

    # Train and save model
    h_params = {
        'max_depth': [3],
        'learning_rate': [0.1],
        'n_estimators': [100]
    }
    train.train(h_params)
    train.dump_models([train.best_model], 'test_scaler', None)

    # Check if model file exists
    assert (tmp_path / 'saved_models' / 'test_xgboost_model.pkl').exists()
