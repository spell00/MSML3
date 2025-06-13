import os
import pytest
import pandas as pd
import numpy as np
from msml.preprocess.make_tensors_sparse_ms2_unsup_transductive import MakeTensorsMultiprocess
from unittest.mock import Mock, patch


@pytest.fixture
def sample_tsv_data():
    """Create sample TSV data for testing"""
    data = {
        'min_parent_mz': [100.0, 100.0, 100.0, 200.0, 200.0],
        'rt_bin': [1.0, 2.0, 3.0, 1.0, 2.0],
        'mz_bin': [1.0, 2.0, 3.0, 1.0, 2.0],
        'bin_intensity': [1.0, 2.0, 3.0, 4.0, 5.0],
        'max_parent_mz': [101.0, 101.0, 101.0, 201.0, 201.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_args():
    """Create mock arguments for testing"""
    args = Mock()
    args.is_sparse = False
    args.save = False
    args.test_run = False
    args.log2 = None
    args.find_peaks = False
    args.lowess = False
    args.n_samples = -1
    args.run_name = 'test_run'
    args.save3d = False
    return args


@pytest.fixture
def bins_config():
    """Create bins configuration for testing"""
    return {
        'mz_shift': False,
        'rt_shift': False,
        'mz_bin_post': 1.0,
        'rt_bin_post': 1.0,
        'mz_rounding': 0,
        'rt_rounding': 0,
        'mz_bin': 1.0,
        'rt_bin': 1.0
    }


def test_init(tmp_path, mock_args, bins_config):
    """Test initialization of MakeTensorsMultiprocess"""
    tsv_list = ['test1.tsv', 'test2.tsv']
    labels_list = ['label1', 'label2']

    processor = MakeTensorsMultiprocess(tsv_list, labels_list, bins_config, str(tmp_path), mock_args)

    assert processor.tsv_list == tsv_list
    assert processor.labels_list == labels_list
    assert processor.bins == bins_config
    assert processor.path == str(tmp_path)
    assert processor.args == mock_args
    assert processor.is_sparse == mock_args.is_sparse
    assert processor.save == mock_args.save
    assert processor.test_run == mock_args.test_run


@patch('pandas.read_csv')
def test_process(mock_read_csv, tmp_path, mock_args, bins_config, sample_tsv_data):
    """Test the process method with sample data"""
    # Setup
    mock_read_csv.return_value = sample_tsv_data
    tsv_list = ['test.tsv']
    labels_list = ['test_label']

    processor = MakeTensorsMultiprocess(tsv_list, labels_list, bins_config, str(tmp_path), mock_args)

    # Execute
    result, keys, label = processor.process(0)

    # Assert
    assert isinstance(result, dict)
    assert len(keys) == 2  # Should have 2 unique min_parent_mz values
    assert label == 'test_label'

    # Check if the data was processed correctly
    for min_parent in result:
        assert isinstance(result[min_parent], pd.DataFrame)
        assert result[min_parent].shape[0] > 0
        assert result[min_parent].shape[1] > 0


def test_n_samples(tmp_path, mock_args, bins_config):
    """Test the n_samples method"""
    tsv_list = ['test1.tsv', 'test2.tsv', 'test3.tsv']
    labels_list = ['label1', 'label2', 'label3']

    processor = MakeTensorsMultiprocess(tsv_list, labels_list, bins_config, str(tmp_path), mock_args)

    assert processor.n_samples() == 3


def test_save_images_and_csv(tmp_path, mock_args, bins_config):
    """Test the save_images_and_csv method"""
    # Create sample data
    data = pd.DataFrame(np.random.rand(10, 10))
    df = np.random.rand(10, 10)
    label = 'test_label'

    processor = MakeTensorsMultiprocess([], [], bins_config, str(tmp_path), mock_args)

    # Execute
    processor.save_images_and_csv(data, df, label)

    # Check if files were created
    assert os.path.exists(os.path.join(str(tmp_path), mock_args.run_name, 'csv', f'{label}.csv'))
    assert os.path.exists(os.path.join(str(tmp_path), mock_args.run_name, 'images', f'{label}.png'))


def test_save_images_and_csv3d(tmp_path, mock_args, bins_config):
    """Test the save_images_and_csv3d method"""
    # Create sample data
    data = pd.DataFrame(np.random.rand(10, 10))
    df = np.random.rand(10, 10)
    label = 'test_label'
    i = 1

    processor = MakeTensorsMultiprocess([], [], bins_config, str(tmp_path), mock_args)

    # Execute
    processor.save_images_and_csv3d(data, df, label, i)

    # Check if files were created
    assert os.path.exists(os.path.join(str(tmp_path), mock_args.run_name, 'csv3d', label, f'{label}_{i}.csv'))
    assert os.path.exists(os.path.join(str(tmp_path), mock_args.run_name, 'images3d', label, f'{label}_{i}.png'))
