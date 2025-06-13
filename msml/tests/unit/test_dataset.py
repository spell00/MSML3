import pytest
import numpy as np
import pandas as pd
from msml.ml.dataset import (
    blocks,
    read_csv,
    read_csv_low_ram,
    get_data_infer,
    get_data_all
)


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing"""
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'label': np.random.randint(0, 2, 100)
    })
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_blocks(sample_csv_file):
    """Test file block reading"""
    block_size = 65536
    for block in blocks(sample_csv_file, block_size):
        assert len(block) <= block_size
        assert isinstance(block, bytes)


def test_read_csv(sample_csv_file):
    """Test CSV reading functionality"""
    num_rows = 50
    n_cols = 2
    data = read_csv(sample_csv_file, num_rows, n_cols)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] <= num_rows
    assert data.shape[1] <= n_cols


def test_read_csv_low_ram(sample_csv_file):
    """Test low memory CSV reading"""
    num_rows = 50
    n_cols = 2
    data = read_csv_low_ram(sample_csv_file, num_rows, n_cols)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] <= num_rows
    assert data.shape[1] <= n_cols


def test_get_data_infer(tmp_path):
    """Test data loading for inference"""
    # Create sample data directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample data files
    for i in range(3):
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100)
        })
        df.to_csv(data_dir / f"sample_{i}.csv", index=False)

    args = type('Args', (), {
        'data_path': str(data_dir),
        'seed': 42
    })

    data = get_data_infer(str(data_dir), args)
    assert isinstance(data, dict)
    assert 'inputs' in data
    assert 'names' in data


def test_get_data_all(tmp_path):
    """Test complete data loading"""
    # Create sample data directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create sample data files
    for i in range(3):
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'label': np.random.randint(0, 2, 100)
        })
        df.to_csv(data_dir / f"sample_{i}.csv", index=False)

    args = type('Args', (), {
        'data_path': str(data_dir),
        'seed': 42
    })

    data = get_data_all(str(data_dir), args)
    assert isinstance(data, dict)
    assert 'inputs' in data
    assert 'labels' in data
    assert 'batches' in data
    assert 'names' in data


def test_read_csv_edge_cases(tmp_path):
    """Test CSV reading edge cases"""
    # Test with empty file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    data = read_csv(empty_file, 100, 100)
    assert isinstance(data, pd.DataFrame)
    assert data.empty

    # Test with file containing only header
    header_file = tmp_path / "header.csv"
    pd.DataFrame(columns=['col1', 'col2']).to_csv(header_file, index=False)
    data = read_csv(header_file, 100, 100)
    assert isinstance(data, pd.DataFrame)
    assert data.empty


def test_read_csv_low_ram_edge_cases(tmp_path):
    """Test low memory CSV reading edge cases"""
    # Test with empty file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    data = read_csv_low_ram(empty_file, 100, 100)
    assert isinstance(data, pd.DataFrame)
    assert data.empty

    # Test with file containing only header
    header_file = tmp_path / "header.csv"
    pd.DataFrame(columns=['col1', 'col2']).to_csv(header_file, index=False)
    data = read_csv_low_ram(header_file, 100, 100)
    assert isinstance(data, pd.DataFrame)
    assert data.empty


def test_get_data_infer_edge_cases(tmp_path):
    """Test data loading for inference edge cases"""
    # Test with empty directory
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    args = type('Args', (), {
        'data_path': str(empty_dir),
        'seed': 42
    })

    data = get_data_infer(str(empty_dir), args)
    assert isinstance(data, dict)
    assert 'inputs' in data
    assert len(data['inputs']) == 0


def test_get_data_all_edge_cases(tmp_path):
    """Test complete data loading edge cases"""
    # Test with empty directory
    empty_dir = tmp_path / "empty_data"
    empty_dir.mkdir()

    args = type('Args', (), {
        'data_path': str(empty_dir),
        'seed': 42
    })

    data = get_data_all(str(empty_dir), args)
    assert isinstance(data, dict)
    assert 'inputs' in data
    assert 'labels' in data
    assert len(data['inputs']) == 0
