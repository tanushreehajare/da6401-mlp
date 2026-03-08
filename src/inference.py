"""
Inference Script
Load a trained model and evaluate it on the test set.
"""
import argparse

import numpy as np
import wandb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ann.activations import Softmax
from ann.neural_network import NeuralNetwork


# ---------------------------------------------------------------------------
# CLI  (same arguments as train.py; best-model defaults pre-filled)
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on a trained model")

    # Weights & Biases project
    parser.add_argument("-wp", "--wandb_project", type=str,
                        default="sweeps-mlp-last")

    # Model path
    parser.add_argument("--model_path", type=str, default="src/best_model.npy",
                        help="Relative path to saved .npy weights")

    # Dataset
    parser.add_argument("-d",   "--dataset",       type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])

    # Best-model architecture defaults (update after training)
    parser.add_argument("-e",   "--epochs",        type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-lr",  "--learning_rate", type=float, default=0.001)
    parser.add_argument("-o",   "--optimizer",     type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-sz",  "--hidden_size",   nargs="+",  type=int,
                        default=[128, 128])
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=2)
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi",  "--weight_init",   type=str,   default="xavier",
                        choices=["xavier", "zeros", "random"])
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_name):
    if dataset_name == "mnist":
        dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    elif dataset_name == "fashion_mnist":
        dataset = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="liac-arff")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = dataset.data.astype(np.float64) / 255.0
    y = dataset.target.astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=10_000, random_state=42)

    y_test_onehot = np.eye(10)[y_test]
    return X_test, y_test, y_test_onehot


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, args):
    """Reconstruct the model architecture and load saved weights."""
    model = NeuralNetwork(args)
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, y_test_onehot):
    """
    Run inference and compute metrics.

    Returns a dict with keys: logits, loss, accuracy, precision, recall, f1.
    """
    logits = model.forward(X_test)   # returns logits; model.probs updated
    y_pred = np.argmax(model.probs, axis=1)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="macro", zero_division=0)
    loss      = model.loss_fn.forward(y_test_onehot, logits)

    # ---- W&B: confusion matrix ----
    cm = wandb.plot.confusion_matrix(
        y_true=y_test,
        preds=y_pred,
        class_names=[str(i) for i in range(10)],
    )
    wandb.log({"confusion_matrix": cm})

    # ---- W&B: misclassified examples ----
    mis_idx = np.where(y_pred != y_test)[0][:10]
    table = wandb.Table(columns=["image", "true", "pred"])
    for idx in mis_idx:
        img = X_test[idx].reshape(28, 28)
        table.add_data(wandb.Image(img), int(y_test[idx]), int(y_pred[idx]))
    wandb.log({"misclassified_examples": table})

    # ---- W&B: scalar metrics ----
    wandb.log({
        "test_loss":      loss,
        "test_accuracy":  accuracy,
        "test_precision": precision,
        "test_recall":    recall,
        "test_f1":        f1,
    })

    return {
        "logits":    logits,
        "loss":      loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project)

    X_test, y_test, y_test_onehot = load_data(args.dataset)
    model   = load_model(args.model_path, args)
    results = evaluate_model(model, X_test, y_test, y_test_onehot)

    print("\n===== Evaluation Results =====")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")

    return results


if __name__ == "__main__":
    main()