import pytest
import numpy as np
import pandas as pd
from msml.ml.train import (
    get_data,
    get_model,
    get_optimizer,
    get_scheduler,
    get_criterion,
    get_device,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


@pytest.fixture
def sample_training_data():
    """Create sample data for training testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    data = {
        'train': {
            'inputs': pd.DataFrame(np.random.randn(n_samples, n_features)),
            'labels': np.array(['class1'] * 50 + ['class2'] * 50)
        },
        'val': {
            'inputs': pd.DataFrame(np.random.randn(20, n_features)),
            'labels': np.array(['class1'] * 10 + ['class2'] * 10)
        }
    }
    return data


def test_get_data(sample_training_data):
    """Test data loading"""
    args = type('Args', (), {
        'data_path': '/path/to/data',
        'batch_size': 32
    })
    
    train_loader, val_loader = get_data(args)
    assert train_loader is not None
    assert val_loader is not None


def test_get_model():
    """Test model initialization"""
    args = type('Args', (), {
        'model_name': 'xgboost',
        'input_size': 50,
        'hidden_size': 100,
        'output_size': 2
    })
    
    model = get_model(args)
    assert model is not None


def test_get_optimizer():
    """Test optimizer initialization"""
    model = type('Model', (), {
        'parameters': lambda: [np.random.randn(10)]
    })
    args = type('Args', (), {
        'learning_rate': 0.001,
        'weight_decay': 0.0001
    })
    
    optimizer = get_optimizer(model, args)
    assert optimizer is not None


def test_get_scheduler():
    """Test learning rate scheduler initialization"""
    optimizer = type('Optimizer', (), {})
    args = type('Args', (), {
        'scheduler': 'cosine',
        'num_epochs': 10
    })
    
    scheduler = get_scheduler(optimizer, args)
    assert scheduler is not None


def test_get_criterion():
    """Test loss function initialization"""
    args = type('Args', (), {
        'criterion': 'cross_entropy'
    })
    
    criterion = get_criterion(args)
    assert criterion is not None


def test_get_device():
    """Test device selection"""
    args = type('Args', (), {
        'device': 'cuda'
    })
    
    device = get_device(args)
    assert device is not None


def test_train_model(sample_training_data):
    """Test model training"""
    model = type('Model', (), {
        'train': lambda: None,
        'parameters': lambda: [np.random.randn(10)]
    })
    optimizer = type('Optimizer', (), {
        'zero_grad': lambda: None,
        'step': lambda: None
    })
    criterion = type('Criterion', (), {
        '__call__': lambda x, y: 0.5
    })
    device = 'cpu'
    
    train_loader = type('DataLoader', (), {
        '__iter__': lambda: iter([(sample_training_data['train']['inputs'], 
                                 sample_training_data['train']['labels'])])
    })
    
    loss = train_model(model, train_loader, optimizer, criterion, device)
    assert isinstance(loss, float)


def test_evaluate_model(sample_training_data):
    """Test model evaluation"""
    model = type('Model', (), {
        'eval': lambda: None,
        'parameters': lambda: [np.random.randn(10)]
    })
    criterion = type('Criterion', (), {
        '__call__': lambda x, y: 0.5
    })
    device = 'cpu'
    
    val_loader = type('DataLoader', (), {
        '__iter__': lambda: iter([(sample_training_data['val']['inputs'], 
                                 sample_training_data['val']['labels'])])
    })
    
    metrics = evaluate_model(model, val_loader, criterion, device)
    assert isinstance(metrics, dict)
    assert 'loss' in metrics


def test_save_model(tmp_path):
    """Test model saving"""
    model = type('Model', (), {
        'state_dict': lambda: {'layer1': np.random.randn(10)}
    })
    path = tmp_path / "model.pth"
    
    save_model(model, str(path))
    assert path.exists()


def test_load_model(tmp_path):
    """Test model loading"""
    # First save a model
    model = type('Model', (), {
        'state_dict': lambda: {'layer1': np.random.randn(10)}
    })
    path = tmp_path / "model.pth"
    save_model(model, str(path))
    
    # Then load it
    loaded_model = load_model(str(path))
    assert loaded_model is not None


def test_get_data_edge_cases():
    """Test data loading with edge cases"""
    args = type('Args', (), {
        'data_path': '/non/existent/path',
        'batch_size': 32
    })
    
    with pytest.raises(FileNotFoundError):
        get_data(args)


def test_get_model_edge_cases():
    """Test model initialization with edge cases"""
    args = type('Args', (), {
        'model_name': 'non_existent_model',
        'input_size': 50,
        'hidden_size': 100,
        'output_size': 2
    })
    
    with pytest.raises(ValueError):
        get_model(args)


def test_get_optimizer_edge_cases():
    """Test optimizer initialization with edge cases"""
    model = type('Model', (), {
        'parameters': lambda: []
    })
    args = type('Args', (), {
        'learning_rate': -0.001,  # Invalid learning rate
        'weight_decay': 0.0001
    })
    
    with pytest.raises(ValueError):
        get_optimizer(model, args)


def test_get_scheduler_edge_cases():
    """Test scheduler initialization with edge cases"""
    optimizer = type('Optimizer', (), {})
    args = type('Args', (), {
        'scheduler': 'non_existent_scheduler',
        'num_epochs': 10
    })
    
    with pytest.raises(ValueError):
        get_scheduler(optimizer, args)


def test_get_criterion_edge_cases():
    """Test criterion initialization with edge cases"""
    args = type('Args', (), {
        'criterion': 'non_existent_criterion'
    })
    
    with pytest.raises(ValueError):
        get_criterion(args)


def test_train_model_edge_cases():
    """Test model training with edge cases"""
    model = type('Model', (), {
        'train': lambda: None,
        'parameters': lambda: []
    })
    optimizer = type('Optimizer', (), {
        'zero_grad': lambda: None,
        'step': lambda: None
    })
    criterion = type('Criterion', (), {
        '__call__': lambda x, y: 0.5
    })
    device = 'cpu'
    
    # Empty data loader
    train_loader = type('DataLoader', (), {
        '__iter__': lambda: iter([])
    })
    
    with pytest.raises(ValueError):
        train_model(model, train_loader, optimizer, criterion, device)


def test_evaluate_model_edge_cases():
    """Test model evaluation with edge cases"""
    model = type('Model', (), {
        'eval': lambda: None,
        'parameters': lambda: []
    })
    criterion = type('Criterion', (), {
        '__call__': lambda x, y: 0.5
    })
    device = 'cpu'
    
    # Empty data loader
    val_loader = type('DataLoader', (), {
        '__iter__': lambda: iter([])
    })
    
    with pytest.raises(ValueError):
        evaluate_model(model, val_loader, criterion, device)


def test_save_model_edge_cases(tmp_path):
    """Test model saving with edge cases"""
    model = type('Model', (), {
        'state_dict': lambda: {}
    })
    path = tmp_path / "non_existent_dir" / "model.pth"
    
    with pytest.raises(FileNotFoundError):
        save_model(model, str(path))


def test_load_model_edge_cases(tmp_path):
    """Test model loading with edge cases"""
    path = tmp_path / "non_existent_model.pth"
    
    with pytest.raises(FileNotFoundError):
        load_model(str(path))
