# README for `run_hlm.sh`
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
