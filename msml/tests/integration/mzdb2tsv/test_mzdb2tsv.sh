#!/bin/bash

# Test setup
TEST_DIR='resources/test_files'

# Test variables
MZ_BIN=10
RT_BIN=10
SPD=200
MS_LEVEL=2
EXPERIMENT="test_experiment"
SPLIT_DATA=1

# Store original directory
ORIGINAL_DIR=$(pwd)

# Change to mzdb2tsv directory
# cd msml/mzdb2tsv || exit 1

# Run the script
echo msml/preprocess/mzdb2tsv.sh "$MZ_BIN" "$RT_BIN" "$SPD" "$MS_LEVEL" "$EXPERIMENT" "$SPLIT_DATA" "$TEST_DIR"
bash msml/preprocess/mzdb2tsv.sh "$MZ_BIN" "$RT_BIN" "$SPD" "$MS_LEVEL" "$EXPERIMENT" "$SPLIT_DATA" "$TEST_DIR"

# Check if output directories were created
if [ ! -d "$TEST_DIR/test_experiment/tsv/mz$MZ_BIN/rt$RT_BIN/${SPD}spd/ms$MS_LEVEL/test" ]; then
    echo "Test failed: Output directory was not created"
    exit 1
fi

# Check if tsv files were created
TSV_COUNT=$(find "$TEST_DIR/test_experiment/tsv" -name "*.tsv" | wc -l)
if [ "$TSV_COUNT" -eq 0 ]; then
    echo "Test failed: No tsv files were created"
    exit 1
fi

echo "All tests passed!"
echo $(pwd)
# cd ../..