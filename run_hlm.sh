#!/bin/bash

# Array of batch sizes
batch_sizes=(8 16 32)
lrs=(1e-6 3e-6 5e-6 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3 3e-3 5e-3)
# Arrays of encoder and decoder models
# encoder_models=("google-bert/bert-base-multilingual-uncased" "FacebookAI/xlm-roberta-base")
# decoder_models=("google/gemma-2-2b" "meta-llama/Llama-3.2-3B" "sarvamai/sarvam-1")
encoder_models=("FacebookAI/xlm-roberta-base")
decoder_models=("google/gemma-2-2b")
for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
        for encoder_model in "${encoder_models[@]}"; do
            for decoder_model in "${decoder_models[@]}"; do
                # Replace slashes with underscores for result_dir
                encoder_name=$(echo "$encoder_model" | tr '/' '_')
                decoder_name=$(echo "$decoder_model" | tr '/' '_')
                result_dir="output/${encoder_name}__${decoder_name}"
                mkdir -p "$result_dir"
                echo "Running HLM_Batch.py with batch size: $batch_size, learning rate: $lr, encoder: $encoder_model, decoder: $decoder_model"
                python3 HLM_Batch.py --batch_size "$batch_size" --learning_rate "$lr" --num_epochs 3 --max_length 256 --encoder_model "$encoder_model" --decoder_model "$decoder_model" --result_dir "$result_dir" --data_dir engtel/
            done
        done
    done
done
