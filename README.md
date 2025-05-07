# README
Executes the HLM Batch script using the batch sizes defined in the `batch_sizes` array and saves the resulting outputs in the designated `results` directory.

### Install Requirements
Before running the script, ensure you have installed the required dependencies. Use the following command to install them from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
## Usage

### Step 1: Make the Script Executable
Before running the script, you need to grant execute permissions. Use the following command:
```bash
chmod +x run_hlm.sh
```

### Step 2: Run the Script
Execute the script using:
```bash
./run_hlm.sh
```
### Step 3: Modify Batch Size
You can change the batch size by editing the `batch_sizes` array in the `run_hlm.sh` script. Open the script and locate the following line:
```bash
batch_sizes=(...)
```
Update the values in the array as needed to adjust the batch size.

## Troubleshooting
- If you encounter a "Permission denied" error, ensure you have run the `chmod +x` command.
- Check the script for any hardcoded paths and update them if necessary.
- If the script doesn't run, ensure it uses Unix-style line endings. Convert it using the `dos2unix` command:
    ```bash
    dos2unix run_hlm.sh
    ```

-----
# HLM_Batch.py

`HLM_Batch.py` is a Python script designed to run training with configurable parameters. Below are the arguments it accepts and their default values:

## Arguments

1. **`--batch_size`**  
    - Description: Batch size for training.  
    - Default: `4`

2. **`--learning_rate`**  
    - Description: Learning rate for training.  
    - Default: `0.0005`

3. **`--num_epochs`**  
    - Description: Number of epochs for training.  
    - Default: `3`

4. **`--max_length`**  
    - Description: Maximum sequence length for tokenization.  
    - Default: `128`

5. **`--encoder_model`**  
    - Description: Encoder model to use (e.g., `bert`, `roberta`).  
    - Default: `bert-base-multilingual-uncased`

6. **`--decoder_model`**  
    - Description: Decoder model to use (e.g., `llama 3.2`, `gpt`).  
    - Default: `meta-llama/Llama-3.2-3B`

7. **`--result_dir`**  
    - Description: Directory to save results.  
    - Default: `results`

8. **`--data_dir`**  
    - Description: Directory containing the dataset files (train, test, and validation).  
    - Default: `engtel`

9. **`--gpu`**  
    - Description: GPU device ID to use.  
    - Default: `0`

## Usage Example

```bash
python HLM_Batch.py --batch_size 8 --learning_rate 0.001 --num_epochs 5 --max_length 256 --encoder_model bert-base-uncased --decoder_model gpt-3 --result_dir output/ --data_dir dataset/ --gpu 1
```

Ensure all required dependencies are installed before running the script.
