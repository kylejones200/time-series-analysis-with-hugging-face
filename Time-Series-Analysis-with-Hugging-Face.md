# Time Series Analysis with Hugging Face

**⚠️ Requires**: `transformers`, `datasets`, `torch` libraries

## Introduction

Hugging Face is best known for its natural language processing tools, but its transformer models offer much broader potential. Time series data, like text, is sequential. Transformers are designed to capture complex dependencies across sequences, making them powerful for time series applications such as forecasting, classification, anomaly detection, and imputation.

## Why Use Hugging Face for Time Series?

Hugging Face provides a rich platform for sequence modeling naturally suited to time series problems:

- **Transformer Architecture**: Captures dependencies across long sequences
- **Transfer Learning**: Pre-trained models fine-tune quickly
- **Multivariate Support**: Handles multiple time series types
- **Flexibility**: Compatible with PyTorch and TensorFlow

## Adapting Transformers for Time Series

### Key Modifications

1. **Tokenization**: Convert numerical sequences to string format
2. **Positional Encoding**: Retain temporal order information
3. **Fine-tuning**: Adapt pre-trained models to time series tasks

## Time Series Classification Example

### Installation
```bash
pip install transformers datasets torch
```

### Step 1: Generate Synthetic Data

```python
import numpy as np
import pandas as pd

# Generate synthetic time series dataset
np.random.seed(42)
n_samples = 100
n_timestamps = 50

# Create time series with two classes
class_0 = np.random.normal(0, 1, (n_samples // 2, n_timestamps))
class_1 = np.random.normal(2, 1, (n_samples // 2, n_timestamps))

# Combine data and labels
X = np.vstack((class_0, class_1))
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Convert to DataFrame
df = pd.DataFrame(X)
df['label'] = y
print(df.head())
```

### Step 2: Tokenize the Data

```python
from transformers import AutoTokenizer

# Use a tokenizer to prepare data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_time_series(series):
    series_str = " ".join(map(str, series))
    return tokenizer(series_str, truncation=True, padding="max_length", 
                    max_length=128, return_tensors="pt")

# Tokenize the dataset
tokens = [tokenize_time_series(row[:-1].values) for _, row in df.iterrows()]
labels = df['label'].values
```

### Step 3: Fine-Tune a Pre-Trained Model

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Split data into train and test sets
train_tokens, test_tokens, train_labels, test_labels = train_test_split(
    tokens, labels, test_size=0.2, random_state=42)

# Create dataset objects
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {key: val.squeeze() for key, val in self.tokens[idx].items()}, self.labels[idx]

train_dataset = TimeSeriesDataset(train_tokens, train_labels)
test_dataset = TimeSeriesDataset(test_tokens, test_labels)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()
```

### Step 4: Evaluate the Model

```python
from sklearn.metrics import accuracy_score

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Test Accuracy: {accuracy:.2f}")
```

## Applications

### Forecasting
Treat future as masked portion of input, model learns to predict missing values.

### Anomaly Detection
Train to reconstruct normal patterns, deviations indicate anomalies.

### Imputation
Mask known values during training, model predicts based on surrounding sequence.

## Advantages

- **Transfer Learning**: Pre-trained models carry useful representations
- **Scalability**: Handle large datasets and long sequences
- **Flexibility**: Support forecasting, classification, anomaly detection

## Challenges

- **Token Limits**: Long series may need truncation or batching
- **Computational Cost**: Requires GPUs/TPUs for training
- **Data Representation**: Numerical to text conversion may reduce precision

## Key Takeaways

- Transformers extend beyond text to time series analysis
- Hugging Face provides infrastructure for experimentation and deployment
- Pre-trained models fine-tune quickly to new domains
- Effective for classification, forecasting, anomaly detection

## Resources

- **Hugging Face Documentation**: https://huggingface.co/docs
- **Transformers Library**: https://github.com/huggingface/transformers
- **Time Series Datasets**: https://huggingface.co/datasets

---

**Note**: This article demonstrates concepts. Production use requires careful hyperparameter tuning, validation, and computational resources.
