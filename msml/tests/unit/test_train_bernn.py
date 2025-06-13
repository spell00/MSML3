import pytest
import numpy as np
import pandas as pd
from msml.ml.train_bernn import Train_bernn


@pytest.fixture
def sample_bernn_data():
    """Create sample data for BERNN training"""
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


def test_bernn_initialization(sample_bernn_data):
    """Test BERNN training class initialization"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )
    assert train.name == 'test_bernn'


def test_bernn_get_data(sample_bernn_data):
    """Test BERNN data preparation"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )

    all_data = sample_bernn_data
    h = 42  # random seed

    train_data, valid_data, test_data = train.get_data(all_data, h)
    assert len(train_data) > 0
    assert len(valid_data) > 0
    assert len(test_data) > 0


def test_bernn_train_bernn(sample_bernn_data):
    """Test BERNN training process"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )

    all_data = sample_bernn_data
    h = 42
    params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    }

    train.train_bernn(all_data, h, params, None)
    assert hasattr(train, 'model')


def test_bernn_get_ordered_layers(sample_bernn_data):
    """Test BERNN layer ordering"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )

    params = {
        'encoder_layers': [64, 32],
        'decoder_layers': [32, 64]
    }

    ordered_layers = train.get_ordered_layers(params)
    assert len(ordered_layers) > 0


def test_bernn_add_encoder_decoder_outputs(sample_bernn_data):
    """Test adding encoder-decoder outputs"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path='./logs',
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )

    lists = {
        'encoder_outputs': [],
        'decoder_outputs': []
    }

    train.add_encoder_decoder_outputs(sample_bernn_data, lists)
    assert len(lists['encoder_outputs']) > 0
    assert len(lists['decoder_outputs']) > 0


def test_bernn_save_load(sample_bernn_data, tmp_path):
    """Test BERNN model saving and loading"""
    train = Train_bernn(
        name='test_bernn',
        model=None,
        data=sample_bernn_data,
        uniques={'labels': ['class1', 'class2'], 'batches': ['batch1', 'batch2']},
        log_path=str(tmp_path),
        args=None,
        logger=None,
        log_neptune=False,
        log_metrics=None
    )

    # Train and save model
    all_data = sample_bernn_data
    h = 42
    params = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10
    }

    train.train_bernn(all_data, h, params, None)
    train.dump_models([train.model], 'test_scaler', None)

    # Check if model file exists
    assert (tmp_path / 'saved_models' / 'test_bernn_model.pkl').exists()
