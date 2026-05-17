import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series classification."""

    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token_dict = {key: val.squeeze() for key, val in self.tokens[idx].items()}
        return (token_dict, self.labels[idx])


def evaluate_model(trainer, test_dataset, test_labels):
    """Evaluate the trained model."""
    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    accuracy = accuracy_score(test_labels, predicted_labels)
    return (predicted_labels, accuracy)


def generate_time_series_data(n_samples=200, n_timesteps=50, seed=42):
    """
    Generate synthetic time series data with two distinct classes.

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    n_timesteps : int
        Length of each time series
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    X : ndarray
        Time series data (n_samples, n_timesteps)
    y : ndarray
        Class labels (n_samples,)
    """
    np.random.seed(seed)
    class_0 = np.random.normal(0, 1, (n_samples // 2, n_timesteps))
    trend = np.linspace(0, 1, n_timesteps)
    class_1 = np.random.normal(2, 1, (n_samples // 2, n_timesteps)) + trend
    X = np.vstack((class_0, class_1))
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return (X, y)


def tokenize_time_series(series, tokenizer, max_length=128):
    """
    Convert time series to tokenized format for transformer models.

    Parameters:
    -----------
    series : array-like
        Time series values
    tokenizer : transformers tokenizer
        Hugging Face tokenizer
    max_length : int
        Maximum sequence length

    Returns:
    --------
    dict
        Tokenized input with input_ids and attention_mask
    """
    series_str = " ".join([f"{val:.4f}" for val in series])
    tokenized = tokenizer(
        series_str,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return tokenized


def train_model(model, train_dataset, eval_dataset, output_dir="./results"):
    """Train the transformer model."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
    trainer.train()
    return trainer


def visualize_results(y_true, y_pred, class_names=None, plot: bool = False):
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    "Create visualization of classification results."
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=class_names,
            yticklabels=class_names,
        )
        axes[0].set_title("Confusion Matrix", fontsize=14)
        axes[0].set_ylabel("True Label", fontsize=12)
        axes[0].set_xlabel("Predicted Label", fontsize=12)
        class_accuracies = []
        for i in range(len(class_names)):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum()
                class_accuracies.append(acc)
            else:
                class_accuracies.append(0)
        axes[1].bar(class_names, class_accuracies, color=["skyblue", "lightcoral"])
        axes[1].set_title("Accuracy by Class", fontsize=14)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_ylim([0, 1])
        for i, acc in enumerate(class_accuracies):
            axes[1].text(i, acc + 0.02, f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig("huggingface_timeseries_classification.png", dpi=300)
        plt.close()


def step_2_prepare_the_time_series_data() -> None:
    "Run complete example."

    logger.info("Time Series Classification with Hugging Face")

    logger.info("\n1. Generating synthetic time series data...")

    X, y = generate_time_series_data(n_samples=200, n_timesteps=50)

    logger.info(f"   Generated {len(X)} samples with {X.shape[1]} timesteps each")

    logger.info(f"   Class distribution: {np.bincount(y)}")

    logger.info("\n2. Tokenizing time series data...")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = [tokenize_time_series(series, tokenizer) for series in X]

    logger.info(f"   Tokenized {len(tokens)} sequences")

    logger.info("\n3. Splitting data into train and test sets...")

    tscv = TimeSeriesSplit(n_splits=5)

    indices = np.arange(len(tokens))

    train_idx, test_idx = list(tscv.split(indices))[-1]

    train_tokens = [tokens[i] for i in train_idx]

    test_tokens = [tokens[i] for i in test_idx]

    train_labels = y[train_idx]

    test_labels = y[test_idx]

    logger.info(f"   Training samples: {len(train_tokens)}")

    logger.info(f"   Test samples: {len(test_tokens)}")

    train_dataset = TimeSeriesDataset(train_tokens, train_labels)

    test_dataset = TimeSeriesDataset(test_tokens, test_labels)

    logger.info("\n4. Loading pre-trained BERT model...")

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    logger.info(f"   Model parameters: {sum((p.numel() for p in model.parameters())):,}")

    logger.info("\n5. Training model...")

    trainer = train_model(model, train_dataset, test_dataset)

    logger.info("\n6. Evaluating model...")

    predicted_labels, accuracy = evaluate_model(trainer, test_dataset, test_labels)

    logger.info(f"\n   Test Accuracy: {accuracy:.4f}")

    logger.info("\n   Classification Report:")

    logger.info(
        classification_report(test_labels, predicted_labels, target_names=["Class 0", "Class 1"])
    )

    logger.info("\n7. Creating visualizations...")

    visualize_results(test_labels, predicted_labels)

    logger.info("   Saved visualization to 'huggingface_timeseries_classification.png'")

    logger.info("=== Example completed successfully! ===")

    logger.info("\nNote: This example demonstrates how to adapt Hugging Face")

    logger.info("transformers for time series classification. The same approach")

    logger.info("can be extended to forecasting, anomaly detection, and imputation tasks.")


def main() -> None:
    step_2_prepare_the_time_series_data()


if __name__ == "__main__":
    main()
