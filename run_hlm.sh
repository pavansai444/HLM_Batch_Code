#!/bin/bash

# Array of batch sizes
batch_sizes=(8 16 32)

# through each batch size and run the Python script

for batch_size in "${batch_sizes[@]}"; do
    echo "Running HLM_Batch.py with batch size: $batch_size"
    python3 HLM_Batch.py --batch_size "$batch_size" --learning_rate 0.0001 --num_epochs 1 --max_length 256 --encoder_model FacebookAI/xlm-roberta-base --decoder_model google/gemma-2-2b --result_dir output/ --data_dir engtel/ 
done
