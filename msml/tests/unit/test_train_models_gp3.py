import pytest
import numpy as np
import pandas as pd
from msml.ml.train_models_gp3 import (
    Train_xgboost,
    Train_bernn,
    TrainAEClassifierHoldout,
    perform_eda,
    create_objective,
    make_args,
    get_batches_infos,
    keep_some_batches,
    keep_some_concs,
    get_args,
    get_path,
    change_data_type,
    get_model
)


@pytest.fixture
def sample_training_data():
    """Create sample data for training models testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    data = {
        'inputs': pd.DataFrame(np.random.randn(n_samples, n_features)),
        'labels': np.array(['class1'] * 50 + ['class2'] * 50),
        'batches': np.array(['batch1'] * 50 + ['batch2'] * 50),
        'concentrations': np.array(['high'] * 50 + ['low'] * 50)
    }
    return data


def test_perform_eda(sample_training_data, tmp_path):
    """Test exploratory data analysis"""
    path = str(tmp_path)
    perform_eda(sample_training_data, path)
    assert (tmp_path / 'eda_report.html').exists()


def test_create_objective(sample_training_data):
    """Test objective function creation"""
    train = type('Train', (), {
        'train': lambda x: {'accuracy': 0.8, 'loss': 0.2}
    })
    args = type('Args', (), {
        'metric': 'accuracy',
        'direction': 'maximize'
    })

    objective = create_objective(train, args)
    assert callable(objective)

    # Test objective function
    trial = type('Trial', (), {
        'suggest_float': lambda x, y, z: 0.1,
        'suggest_int': lambda x, y, z: 10
    })
    result = objective(trial)
    assert isinstance(result, dict)
    assert 'accuracy' in result


def test_make_args():
    """Test argument creation"""
    batches_to_keep = ['batch1', 'batch2']
    concs = ['high', 'low']

    args = make_args(None, batches_to_keep, concs)
    assert isinstance(args, type('Args', (), {}))
    assert hasattr(args, 'batches_to_keep')
    assert hasattr(args, 'concs')


def test_get_batches_infos():
    """Test batch information retrieval"""
    info = get_batches_infos()
    assert isinstance(info, dict)
    assert 'batches' in info
    assert 'concentrations' in info


def test_keep_some_batches(sample_training_data):
    """Test batch filtering"""
    batches_to_keep = ['batch1']
    filtered_data = keep_some_batches(sample_training_data, batches_to_keep)

    assert len(filtered_data['inputs']) < len(sample_training_data['inputs'])
    assert all(batch in batches_to_keep for batch in filtered_data['batches'])


def test_keep_some_concs(sample_training_data):
    """Test concentration filtering"""
    concs = ['high']
    filtered_data = keep_some_concs(sample_training_data, concs)

    assert len(filtered_data['inputs']) < len(sample_training_data['inputs'])
    assert all(conc in concs for conc in filtered_data['concentrations'])


def test_get_args():
    """Test argument retrieval"""
    batches_to_keep = ['batch1', 'batch2']
    args = get_args(batches_to_keep)
    assert isinstance(args, type('Args', (), {}))
    assert hasattr(args, 'data_path')
    assert hasattr(args, 'model_name')


def test_get_path():
    """Test path generation"""
    args = type('Args', (), {
        'data_path': '/path/to/data',
        'model_name': 'test_model'
    })
    exp = 'test_experiment'

    path = get_path(args, exp)
    assert isinstance(path, str)
    assert 'test_model' in path
    assert 'test_experiment' in path


def test_change_data_type(sample_training_data):
    """Test data type conversion"""
    args = type('Args', (), {
        'data_type': 'float32'
    })

    converted_data = change_data_type(sample_training_data, args)
    assert converted_data['inputs'].dtype == np.float32


def test_get_model():
    """Test model initialization"""
    args = type('Args', (), {
        'model_name': 'xgboost'
    })

    model = get_model(args)
    assert model is not None


def test_perform_eda_edge_cases(tmp_path):
    """Test EDA with edge cases"""
    # Test with empty data
    empty_data = {
        'inputs': pd.DataFrame(),
        'labels': np.array([]),
        'batches': np.array([]),
        'concentrations': np.array([])
    }
    path = str(tmp_path)
    perform_eda(empty_data, path)
    assert (tmp_path / 'eda_report.html').exists()


def test_create_objective_edge_cases():
    """Test objective function with edge cases"""
    train = type('Train', (), {
        'train': lambda x: {'accuracy': 0.0, 'loss': 1.0}
    })
    args = type('Args', (), {
        'metric': 'accuracy',
        'direction': 'maximize'
    })

    objective = create_objective(train, args)
    trial = type('Trial', (), {
        'suggest_float': lambda x, y, z: 0.0,
        'suggest_int': lambda x, y, z: 0
    })

    result = objective(trial)
    assert isinstance(result, dict)
    assert 'accuracy' in result
    assert result['accuracy'] == 0.0


def test_keep_some_batches_edge_cases(sample_training_data):
    """Test batch filtering with edge cases"""
    # Test with empty batches list
    filtered_data = keep_some_batches(sample_training_data, [])
    assert len(filtered_data['inputs']) == 0

    # Test with non-existent batch
    filtered_data = keep_some_batches(sample_training_data, ['non_existent'])
    assert len(filtered_data['inputs']) == 0


def test_keep_some_concs_edge_cases(sample_training_data):
    """Test concentration filtering with edge cases"""
    # Test with empty concentrations list
    filtered_data = keep_some_concs(sample_training_data, [])
    assert len(filtered_data['inputs']) == 0

    # Test with non-existent concentration
    filtered_data = keep_some_concs(sample_training_data, ['non_existent'])
    assert len(filtered_data['inputs']) == 0


def test_xgboost_training(sample_training_data):
    """Test XGBoost training process"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = Train_xgboost()
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1


def test_bernn_training(sample_training_data):
    """Test BERNN training process"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = Train_bernn()
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1


def test_ae_classifier_training(sample_training_data):
    """Test AE Classifier training process"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = TrainAEClassifierHoldout()
    model.fit(X, y)
    score = model.score(X, y)
    assert 0 <= score <= 1


def test_model_cross_validation(sample_training_data):
    """Test cross-validation for models"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = Train_xgboost()
    cv_scores = model.cross_validate(X, y, cv=5)
    assert len(cv_scores) == 5
    assert all(0 <= score <= 1 for score in cv_scores)


def test_model_hyperparameter_tuning(sample_training_data):
    """Test hyperparameter tuning for models"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = Train_xgboost()
    best_params = model.tune_hyperparameters(X, y)
    assert isinstance(best_params, dict)
    assert len(best_params) > 0


def test_model_feature_importance(sample_training_data):
    """Test feature importance calculation"""
    X, y = sample_training_data['inputs'], sample_training_data['labels']
    model = Train_xgboost()
    model.fit(X, y)
    importance = model.get_feature_importance()
    assert len(importance) == X.shape[1]
    assert all(imp >= 0 for imp in importance)
