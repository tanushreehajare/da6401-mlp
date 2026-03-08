"""
Main Training Script
Entry point for training neural networks with command-line arguments.
"""
import argparse
import json
import os

import numpy as np
import wandb
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ann.activations import Softmax
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp

BEST_MODEL_PATH = "src/best_model.npy"
BEST_CONFIG_PATH = "src/best_config.json"
BEST_SCORE_PATH = "src/best_score.txt"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST / Fashion-MNIST")

    # Weights & Biases project
    parser.add_argument("-wp", "--wandb_project", type=str,
                        default="sweeps-mlp-last",
                        help="Weights & Biases project name")

    # Dataset & training hyper-parameters
    parser.add_argument("-d",   "--dataset",       type=str,   required=True,
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   required=True)
    parser.add_argument("-b",   "--batch_size",    type=int,   required=True)
    parser.add_argument("-lr",  "--learning_rate", type=float, required=True)
    parser.add_argument("-o",   "--optimizer",     type=str,   required=True,
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-sz",  "--hidden_size",   nargs="+",  type=int, required=True,
                        help="Neuron count per hidden layer (list, length == num_layers)")
    parser.add_argument("-nhl", "--num_layers",    type=int,   required=True,
                        help="Number of hidden layers")
    parser.add_argument("-a",   "--activation",    type=str,   required=True,
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l",   "--loss",          type=str,   required=True,
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-wi",  "--weight_init",   type=str,   required=True,
                        choices=["xavier", "zeros", "random"])
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0,
                        help="L2 regularisation coefficient")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=42
    )

    # One-hot encode labels for training and test sets
    y_train_oh = np.eye(10)[y_train]
    y_test_oh  = np.eye(10)[y_test]

    return X_train, X_test, y_train_oh, y_test_oh


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def get_optimizer(name, learning_rate, weight_decay):
    if name == "sgd":
        return SGD(learning_rate, weight_decay)
    elif name == "momentum":
        return Momentum(learning_rate=learning_rate, weight_decay=weight_decay)
    elif name == "nag":
        return NAG(learning_rate, weight_decay=weight_decay)
    elif name == "rmsprop":
        return RMSProp(learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ---------------------------------------------------------------------------
# W&B logging helpers
# ---------------------------------------------------------------------------

def log_class_samples(X, y):
    """Log 5 sample images per class (50 images total) to a W&B Table."""
    table = wandb.Table(columns=["image", "label"])
    y_labels = np.argmax(y, axis=1)
    for cls in range(10):
        indices = np.where(y_labels == cls)[0][:5]
        for idx in indices:
            img = X[idx].reshape(28, 28)
            table.add_data(wandb.Image(img), cls)
    wandb.log({"class_samples": table})


# ---------------------------------------------------------------------------
# Training entry point (reusable by sweep)
# ---------------------------------------------------------------------------

def main_with_args(args):
    X_train, X_test, y_train, y_test = load_data(args.dataset)

    log_class_samples(X_train, y_train)

    model = NeuralNetwork(args)

    # Attach optimizer and override update_weights
    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)
    model.optimizer = optimizer

    def update_with_optimizer():
        model.optimizer.update(model.layers)

    model.update_weights = update_with_optimizer

    model.train(X_train, y_train, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size)

    # Final evaluation
    logits = model.forward(X_test)
    y_pred  = np.argmax(model.probs, axis=1)
    y_true  = np.argmax(y_test, axis=1)
    test_accuracy = float(np.mean(y_pred == y_true))
    test_f1       = float(f1_score(y_true, y_pred, average="macro"))

    wandb.log({"test_accuracy": test_accuracy, "test_f1": test_f1})

    return model, test_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    wandb.init(project=args.wandb_project, config=vars(args))

    # ------- Load data -------
    X_train, X_test, y_train, y_test = load_data(args.dataset)

    # ------- Q2.1 — Log class samples -------
    log_class_samples(X_train, y_train)

    # ------- Build model -------
    model = NeuralNetwork(args)

    # ------- Attach optimizer -------
    optimizer = get_optimizer(args.optimizer, args.learning_rate, args.weight_decay)
    model.optimizer = optimizer

    def update_with_optimizer():
        model.optimizer.update(model.layers)

    model.update_weights = update_with_optimizer

    # ------- Train -------
    model.train(X_train, y_train, X_test, y_test,
                epochs=args.epochs, batch_size=args.batch_size)

    # ------- Evaluate -------
    logits        = model.forward(X_test)   # returns logits; self.probs updated
    y_pred         = np.argmax(model.probs, axis=1)
    y_true         = np.argmax(y_test, axis=1)
    test_accuracy  = float(np.mean(y_pred == y_true))
    test_f1        = float(f1_score(y_true, y_pred, average="macro"))

    wandb.log({"test_accuracy": test_accuracy, "test_f1": test_f1})
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")

    # ------- Save best model -------
    os.makedirs("src", exist_ok=True)

    best_score = -1.0
    if os.path.exists(BEST_SCORE_PATH):
        with open(BEST_SCORE_PATH, "r") as fh:
            best_score = float(fh.read().strip())

    if test_f1 > best_score:
        print(f"New best model (F1 = {test_f1:.4f}) — saving …")
        np.save(BEST_MODEL_PATH, model.get_weights())
        with open(BEST_SCORE_PATH, "w") as fh:
            fh.write(str(test_f1))
        with open(BEST_CONFIG_PATH, "w") as fh:
            json.dump(vars(args), fh, indent=4)
    else:
        print(f"Model F1 ({test_f1:.4f}) did not beat current best ({best_score:.4f}).")


if __name__ == "__main__":
    main()