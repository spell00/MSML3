import pytest
import numpy as np
from scipy.sparse import csr_matrix
from msml.preprocess.utils import (
    crop_data,
    adjust_tensors,
    delete_rows_csr
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = np.random.rand(10, 10)
    columns = [f"peak_{i}_{j}" for i in range(10) for j in range(10)]
    return data, columns


@pytest.fixture
def sample_sparse_matrix():
    """Create sample sparse matrix for testing"""
    return csr_matrix(np.random.rand(10, 10))


def test_crop_data(sample_data):
    """Test cropping data function"""
    data, columns = sample_data
    args = type('Args', (), {
        'min_mz': 2,
        'max_mz': 7,
        'min_rt': 2,
        'max_rt': 7
    })
    
    cropped_data, cropped_columns = crop_data(data, columns, args)
    assert cropped_data.shape[1] == len(cropped_columns)
    assert all('_2_' in col or '_3_' in col or '_4_' in col or '_5_' in col or '_6_' in col or '_7_' in col 
              for col in cropped_columns)


def test_adjust_tensors():
    """Test tensor adjustment function"""
    # Create sample matrices
    matrices = [{
        'key1': csr_matrix(np.random.rand(5, 5)),
        'key2': csr_matrix(np.random.rand(5, 5))
    }]
    list_matrices = [matrices]
    
    max_features = {
        'parents': 5,
        'max_rt': 8,
        'max_mz': 8
    }
    
    args_dict = type('Args', (), {
        'mz_bin': 0.1,
        'rt_bin': 0.1,
        'mz_bin_post': 0.1,
        'rt_bin_post': 0.1
    })
    
    adjusted = adjust_tensors(list_matrices, max_features, args_dict)
    assert len(adjusted) == 1
    assert adjusted[0][0].shape[1] == max_features['max_rt'] * max_features['max_mz']


def test_delete_rows_csr(sample_sparse_matrix):
    """Test deleting rows from CSR matrix"""
    rows_to_delete = [0, 2, 4]
    result = delete_rows_csr(sample_sparse_matrix, rows_to_delete)
    assert result.shape[0] == sample_sparse_matrix.shape[0] - len(rows_to_delete)
    assert result.shape[1] == sample_sparse_matrix.shape[1]


def test_crop_data_edge_cases():
    """Test crop_data with edge cases"""
    # Test with empty data
    empty_data = np.array([])
    empty_columns = []
    args = type('Args', (), {
        'min_mz': 2,
        'max_mz': 7,
        'min_rt': 2,
        'max_rt': 7
    })
    with pytest.raises(IndexError):
        crop_data(empty_data, empty_columns, args)

    # Test with invalid mz/rt ranges
    data = np.random.rand(10, 10)
    columns = [f"peak_{i}_{j}" for i in range(10) for j in range(10)]
    args = type('Args', (), {
        'min_mz': 20,  # Outside range
        'max_mz': 30,  # Outside range
        'min_rt': 20,  # Outside range
        'max_rt': 30   # Outside range
    })
    cropped_data, cropped_columns = crop_data(data, columns, args)
    assert len(cropped_columns) == 0
    assert cropped_data.shape[1] == 0


def test_adjust_tensors_edge_cases():
    """Test adjust_tensors with edge cases"""
    # Test with empty matrices
    empty_matrices = []
    max_features = {
        'parents': 5,
        'max_rt': 8,
        'max_mz': 8
    }
    args_dict = type('Args', (), {
        'mz_bin': 0.1,
        'rt_bin': 0.1,
        'mz_bin_post': 0.1,
        'rt_bin_post': 0.1
    })
    adjusted = adjust_tensors([empty_matrices], max_features, args_dict)
    assert len(adjusted) == 1
    assert len(adjusted[0]) == 0


def test_delete_rows_csr_edge_cases(sample_sparse_matrix):
    """Test delete_rows_csr with edge cases"""
    # Test with empty list of rows to delete
    result = delete_rows_csr(sample_sparse_matrix, [])
    assert result.shape == sample_sparse_matrix.shape

    # Test with invalid row indices
    with pytest.raises(IndexError):
        delete_rows_csr(sample_sparse_matrix, [100])

    # Test with non-CSR matrix
    with pytest.raises(ValueError):
        delete_rows_csr(np.random.rand(10, 10), [0]) 