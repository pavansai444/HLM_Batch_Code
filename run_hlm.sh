#!/bin/bash

# Array of batch sizes
batch_sizes=(8 16 32 50 100)

# through each batch size and run the Python script
for batch_size in "${batch_sizes[@]}"; do
    echo "Running HLM_Batch.py with batch size: $batch_size"
    python3 HLM_Batch.py --batch_size "$batch_size"
done