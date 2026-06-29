#!/usr/bin/env bash
set -e

# Change directory to script's location
cd "$(dirname "$0")"

# Default input file
INPUT_LOG=${1:-/tmp/parsed_serialized.txt}
DATA_DIR="./data"
ARTIFACTS_DIR="./artifacts"

if [ ! -f "$INPUT_LOG" ]; then
    echo "Error: Input log file '$INPUT_LOG' does not exist."
    echo "Usage: ./run_pipeline.sh [path/to/parsed_serialized.txt]"
    exit 1
fi

echo "=== [Velox Pipeline] Step 1: Preprocessing log file into Parquet ==="
python3 preprocess.py --input "$INPUT_LOG" --output-dir "$DATA_DIR"

echo "=== [Velox Pipeline] Step 2: Training Velox GNN model ==="
python3 train.py --data-dir "$DATA_DIR" --artifacts-dir "$ARTIFACTS_DIR"

echo "=== [Velox Pipeline] Step 3: Evaluating Velox GNN model ==="
python3 eval.py --data-dir "$DATA_DIR" --artifacts-dir "$ARTIFACTS_DIR"

echo "=== [Velox Pipeline] Pipeline executed successfully! ==="
