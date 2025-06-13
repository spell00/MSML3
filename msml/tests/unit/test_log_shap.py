import pytest
import numpy as np
import pandas as pd
import shap
from msml.ml.log_shap import (
    interactions_mean_matrix,
    make_summary_plot,
    make_force_plot,
    make_deep_beeswarm,
    make_decision_plot,
    make_decision_deep,
    make_multioutput_decision_plot,
    make_group_difference_plot,
    make_beeswarm_plot,
    make_heatmap,
    make_heatmap_deep,
    make_barplot,
    make_bar_plot,
    make_dependence_plot,
    log_explainer,
    log_kernel_explainer,
    log_shap
)


@pytest.fixture
def sample_shap_data():
    """Create sample data for SHAP analysis"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = np.random.randint(0, 2, n_samples)

    return X, y


@pytest.fixture
def sample_shap_values(sample_shap_data):
    """Create sample SHAP values"""
    X, _ = sample_shap_data
    return np.random.randn(X.shape[0], X.shape[1])


def test_interactions_mean_matrix(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP interactions mean matrix calculation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'

    mean_matrix = interactions_mean_matrix(sample_shap_values, X, run, group)
    assert mean_matrix.shape == (X.shape[1], X.shape[1])


def test_make_summary_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP summary plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_summary_plot(X, sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_summary_test.png').exists()


def test_make_force_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP force plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_force_plot(X, sample_shap_values, X.columns, group, run, log_path)
    assert (tmp_path / 'shap_force_test.png').exists()


def test_make_deep_beeswarm(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP deep beeswarm plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_deep_beeswarm(X, sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_beeswarm_test.png').exists()


def test_make_decision_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP decision plot creation"""
    X, y = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)
    misclassified = np.random.randint(0, 2, len(y))

    make_decision_plot(X, sample_shap_values, misclassified, X.columns, group, run, log_path)
    assert (tmp_path / 'shap_decision_test.png').exists()


def test_make_decision_deep(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP deep decision plot creation"""
    X, y = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)
    misclassified = np.random.randint(0, 2, len(y))

    make_decision_deep(X, sample_shap_values, misclassified, X.columns, group, run, log_path)
    assert (tmp_path / 'shap_decision_deep_test.png').exists()


def test_make_multioutput_decision_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP multioutput decision plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_multioutput_decision_plot(X, sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_multioutput_decision_test.png').exists()


def test_make_group_difference_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP group difference plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)
    mask = np.random.randint(0, 2, len(X))

    make_group_difference_plot(sample_shap_values, mask, group, run, log_path)
    assert (tmp_path / 'shap_group_diff_test.png').exists()


def test_make_beeswarm_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP beeswarm plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_beeswarm_plot(sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_beeswarm_test.png').exists()


def test_make_heatmap(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP heatmap creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_heatmap(sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_heatmap_test.png').exists()


def test_make_heatmap_deep(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP deep heatmap creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_heatmap_deep(sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_heatmap_deep_test.png').exists()


def test_make_barplot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP barplot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_barplot(X, X.columns, sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_bar_test.png').exists()


def test_make_bar_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP bar plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_bar_plot(X, sample_shap_values, group, run, log_path)
    assert (tmp_path / 'shap_bar_test.png').exists()


def test_make_dependence_plot(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP dependence plot creation"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    make_dependence_plot(X, sample_shap_values, 'feature_0', group, run, log_path)
    assert (tmp_path / 'shap_dependence_test.png').exists()


def test_log_explainer(sample_shap_values, sample_shap_data, tmp_path):
    """Test SHAP explainer logging"""
    X, _ = sample_shap_data
    run = None
    group = 'test'
    args_dict = {
        'inputs': X,
        'values': sample_shap_values
    }

    log_explainer(run, group, args_dict)


def test_log_kernel_explainer(sample_shap_data, tmp_path):
    """Test SHAP kernel explainer logging"""
    X, y = sample_shap_data
    run = None
    group = 'test'
    log_path = str(tmp_path)

    # Create a simple model for testing
    model = shap.KernelExplainer(lambda x: np.zeros(len(x)), X.iloc[:10])

    log_kernel_explainer(model, X, np.zeros(len(X)), y, group, run, ['class1', 'class2'], log_path)


def test_log_shap(sample_shap_data, tmp_path):
    """Test main SHAP logging function"""
    X, y = sample_shap_data
    run = None
    args_dict = {
        'inputs': X,
        'values': np.random.randn(X.shape[0], X.shape[1])
    }

    log_shap(run, args_dict)
