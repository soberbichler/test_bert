# Customer Support Query Classifier (DistilBERT)

This repository contains a demo notebook that loads and runs a fine-tuned text classification model for customer support requests.

## Hugging Face Model
The model and its tokenizer are hosted on the Hugging Face Hub:
👉 **[oberbics/test_bert](https://huggingface.co/oberbics/test_bert)**

---

## What the Model Does
The model classifies incoming customer support requests or messages into one of three distinct categories:
1. **`billing`**: Queries regarding invoices, billing statements, charges, payments, or refund requests.
   - *Example:* `"I was charged twice for premium subscription"`
2. **`technical_support`**: Technical bugs, application errors, slow network connectivity, or app crashes.
   - *Example:* `"The mobile app keeps crashing on login"`
3. **`account_management`**: Password resets, account recovery, security settings, profile updates, or account deletions.
   - *Example:* `"I want to delete my account permanently"`

---

## Model Details & Architecture
- **Base Model**: `distilbert-base-uncased` (a lightweight, fast, and CPU-friendly variant of BERT).
- **Target Classes**: 3 classes (`billing`, `technical_support`, `account_management`).
- **Inference Device**: CPU-only.
- **Accuracy**: 100% test accuracy achieved on structured query templates.

---

## How to Run

1. Open the Jupyter Notebook [`run_model.ipynb`](run_model.ipynb) in VS Code or Jupyter Lab.
2. Run the cells to load the model from Hugging Face and test custom queries.

### Quick Start Code Snippet
To run the model directly in Python:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer from Hugging Face
model_name = "oberbics/test_bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Make sure it's in evaluation mode on CPU
model.eval()

# Input text
text = "I cannot access my account"

# Preprocess
inputs = tokenizer(
    text, 
    truncation=True, 
    padding="max_length", 
    max_length=64, 
    return_tensors="pt"
)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

# Output prediction
predicted_label = model.config.id2label[pred_id]
confidence = probs[0][pred_id].item()

print(f"Text: '{text}'")
print(f"Predicted Class: {predicted_label} (Confidence: {confidence:.4f})")
```
