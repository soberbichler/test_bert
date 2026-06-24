# CPU-Friendly Text Classifier

A simple, CPU-friendly text classification pipeline that generates a fake dataset for customer support requests, fine-tunes a `distilbert-base-uncased` model, and performs inference.

## Project Structure
- `requirements.txt`: Python package dependencies.
- `generate_dataset.py`: Generates mock dataset and splits it into training, validation, and test sets.
- `train.py`: Fine-tunes DistilBERT on CPU, reporting loss and calculating validation/test accuracies.
- `predict.py`: CLI tool to predict the label of a given string using the trained model.
- `upload_to_hub.py`: Utility script to upload the model to Hugging Face.
- `upload_to_github.py`: Utility script to create a GitHub repo and push files.
- `run_model.ipynb`: Jupyter notebook demonstrating how to load and predict using the model.
- `data/`: Directory where generated datasets (`train.csv`, `val.csv`, `test.csv`) are stored.
- `model_output/`: Directory where the trained model and tokenizer config are saved.

## Installation
Ensure you have Python 3.8+ installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

*Note: For maximum speed on standard CPU machines, you can install the CPU-only version of PyTorch:*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## How to Run

### 1. Generate the Dataset
Create the mock dataset with 3 classes (`billing`, `technical_support`, `account_management`):
```bash
python generate_dataset.py
```
This generates:
- `data/train.csv` (70%)
- `data/val.csv` (15%)
- `data/test.csv` (15%)

### 2. Train the Model
Train the model on the CPU for 2 epochs:
```bash
python train.py
```
During training, the script will:
- Print the current loss every 10 steps.
- Print the average training loss and validation accuracy at the end of each epoch.
- Compute and print the final test accuracy after all epochs.
- Save the tokenizer and model configuration to the `model_output/` folder.

### 3. Run Predictions
Perform inference on a text query:
```bash
python predict.py "I cannot access my account"
```
Example Output:
```text
Text: 'I cannot access my account'
Predicted Class: account_management (Confidence: 0.9842)
```

### 4. Upload to Hugging Face
To upload the fine-tuned model directly to your Hugging Face page under `oberbics/test_bert`, run the `upload_to_hub.py` script:

```bash
# Defaults to repository "oberbics/test_bert"
python upload_to_hub.py --token "YOUR_HF_WRITE_TOKEN"

# Or specify a different repo name
python upload_to_hub.py "oberbics/other-repo-name" --token "YOUR_HF_WRITE_TOKEN"
```

Replace `YOUR_HF_WRITE_TOKEN` with a Hugging Face Access Token that has **Write** permissions (generate one in your Hugging Face settings under Settings -> Access Tokens).

### 5. Upload to GitHub
To push all code, scripts, and the Jupyter notebook to a new GitHub repository (`soberbichler/test_bert`), run:

```bash
python upload_to_github.py
```
This script will prompt you for your GitHub Personal Access Token (PAT) with `repo` permissions, automatically create the `test_bert` repository on your GitHub account, configure Git, stage the code, commit, and push it for you. It automatically ignores large weights in `model_output/` to keep your git repo lightweight.


