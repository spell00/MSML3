#!/bin/bash
set -e  # Exit on any error

# Test variables
MZ_BIN=10
RT_BIN=10
SPD=200
MS_LEVEL=2
EXPERIMENT="test_experiment"
SPLIT_DATA=0

# Create test directory structure
TEST_DIR="msml/tests/integration/mzdb2tsv/test_data"
mkdir -p "$TEST_DIR/mzdb/${SPD}spd"
mkdir -p "$TEST_DIR/tsv/mz${MZ_BIN}/rt${RT_BIN}/${SPD}spd/ms${MS_LEVEL}/all"

# Create a dummy mzdb file for testing
echo "Creating test mzdb file..."
touch "$TEST_DIR/mzdb/${SPD}spd/test.mzDB"

# Run the conversion
echo "Running mzdb2tsv conversion..."
cd msml/mzdb2tsv
JAVA_OPTS="-Djava.library.path=$(pwd)" ./amm dia_maps_histogram.sc \
    "../../tests/integration/mzdb2tsv/test_data/mzdb/${SPD}spd/test.mzDB" \
    "$MZ_BIN" "$RT_BIN"

# Check if output was created
echo "Verifying output..."
if [ ! -f "test.tsv" ]; then
    echo "Error: No tsv file was created"
    exit 1
fi

# Check tsv content
if [ ! -s "test.tsv" ]; then
    echo "Error: TSV file is empty"
    exit 1
fi

# Check for required columns
if ! grep -q "mz.*rt.*intensity" "test.tsv"; then
    echo "Error: TSV file does not contain required columns"
    exit 1
fi

echo "All tests passed!"
exit 0 