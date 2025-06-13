import pytest
import pandas as pd
from msml.mzdb2tsv import Mzdb2tsv


@pytest.fixture
def sample_mzdb_data():
    """Create sample mzdb data for testing"""
    return {
        'mz': [100.0, 101.0, 102.0],
        'rt': [1.0, 2.0, 3.0],
        'intensity': [1000.0, 2000.0, 3000.0]
    }


def test_mzdb2tsv_initialization():
    """Test Mzdb2tsv class initialization"""
    converter = Mzdb2tsv()
    assert converter is not None


def test_convert_to_tsv(sample_mzdb_data, tmp_path):
    """Test conversion of mzdb to tsv"""
    converter = Mzdb2tsv()
    input_file = tmp_path / "test.mzdb"
    output_file = tmp_path / "test.tsv"

    # Create a mock mzdb file
    pd.DataFrame(sample_mzdb_data).to_csv(input_file, sep='\t', index=False)

    # Convert to tsv
    converter.convert(input_file, output_file)

    # Check if output file exists and has correct format
    assert output_file.exists()
    result = pd.read_csv(output_file, sep='\t')
    assert all(col in result.columns for col in ['mz', 'rt', 'intensity'])


def test_batch_processing(tmp_path):
    """Test batch processing of multiple mzdb files"""
    converter = Mzdb2tsv()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create multiple mock mzdb files
    for i in range(3):
        data = {
            'mz': [100.0 + i, 101.0 + i],
            'rt': [1.0, 2.0],
            'intensity': [1000.0, 2000.0]
        }
        pd.DataFrame(data).to_csv(input_dir / f"test_{i}.mzdb", sep='\t', index=False)

    # Process all files
    converter.process_directory(input_dir, output_dir)

    # Check if all files were converted
    assert len(list(output_dir.glob("*.tsv"))) == 3


def test_data_validation(sample_mzdb_data):
    """Test data validation in mzdb2tsv"""
    converter = Mzdb2tsv()

    # Test with valid data
    assert converter.validate_data(pd.DataFrame(sample_mzdb_data))

    # Test with invalid data (missing column)
    invalid_data = sample_mzdb_data.copy()
    del invalid_data['mz']
    assert not converter.validate_data(pd.DataFrame(invalid_data))


def test_error_handling(tmp_path):
    """Test error handling in mzdb2tsv"""
    converter = Mzdb2tsv()
    input_file = tmp_path / "nonexistent.mzdb"
    output_file = tmp_path / "output.tsv"

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        converter.convert(input_file, output_file)
