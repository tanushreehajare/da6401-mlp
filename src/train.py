"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import os
import wandb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from ann.activations import Softmax
from ann.neural_network import NeuralNetwork
from ann.optimizers import SGD, Momentum, NAG, RMSProp

import matplotlib.pyplot as plt

import json

BEST_MODEL_PATH = "best_model.npy"
BEST_CONFIG_PATH = "best_config.json"
BEST_SCORE_PATH = "best_score.txt"

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-wp", "--wandb_project", type=str,
                    default="sweeps-mlp-last")

    parser.add_argument("-d","--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("-e", "--epochs", type=int, required=True)

    parser.add_argument("-b", "--batch_size", type=int, required=True)

    parser.add_argument("-lr", "--learning_rate", type=float, required=True)

    parser.add_argument("-o", "--optimizer", type=str, required=True,
                        choices=["sgd", "momentum", "nag",
                                 "rmsprop"])
    
    parser.add_argument("-sz", "--hidden_size", nargs="+", type=int, required=True)

    parser.add_argument("-nhl", "--num_layers", type=int, required=True)

    parser.add_argument("-a", "--activation", type=str, required=True,
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("-l", "--loss", type=str, required=True,
                        choices=["cross_entropy", "mse"])

    parser.add_argument("-wi", "--weight_init", type=str, required=True,
                        choices=["xavier","zeros","random"])

    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    args = parser.parse_args()

    return args

def load_data(dataset_name):

    if dataset_name == "mnist":
        dataset = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
            parser="liac-arff"
        )

    elif dataset_name == "fashion_mnist":
        dataset = fetch_openml(
            "Fashion-MNIST",
            version=1,
            as_frame=False,
            parser="liac-arff"
        )

    else:
        raise ValueError("Invalid dataset")

    X = dataset.data
    y = dataset.target.astype(int)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, X_test, y_train, y_test

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
        raise ValueError("Unsupported optimizer")
    
def log_class_samples(X, y):
    """
    Logs exactly 5 images per class (total 50 images)
    """
    table = wandb.Table(columns=["image", "label"])
    y_labels = np.argmax(y, axis=1)

    for cls in range(10):
        indices = np.where(y_labels == cls)[0][:5]  # exactly 5
        for idx in indices:
            img = X[idx].reshape(28, 28)
            table.add_data(wandb.Image(img), cls)

    wandb.log({"class_samples": table})
    
def main_with_args(args):

    X_train, X_test, y_train, y_test = load_data(args.dataset)

    # ADD THIS LINE
    log_class_samples(X_train, y_train)

    model = NeuralNetwork(args)

    optimizer = get_optimizer(
        args.optimizer,
        args.learning_rate,
        args.weight_decay
    )

    model.optimizer = optimizer

    def update_with_optimizer():
        model.optimizer.update(model.layers)

    model.update_weights = update_with_optimizer

    model.train(X_train, y_train,
                X_test, y_test,
                epochs=args.epochs,
                batch_size=args.batch_size)

    from sklearn.metrics import f1_score

    logits = model.forward(X_test)
    probs = Softmax.forward(logits)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_accuracy = np.mean(y_pred == y_true)
    test_f1 = f1_score(y_true, y_pred, average="macro")

    wandb.log({
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    })

def main():

    args = parse_arguments()

    wandb.init(project=args.wandb_project, config=vars(args))

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    
    # Q2.1 Logging
    log_class_samples(X_train, y_train)

    # Create model
    model = NeuralNetwork(args)

    # Attach optimizer
    optimizer = get_optimizer(args.optimizer,
                               args.learning_rate,
                               args.weight_decay)

    model.optimizer = optimizer   # attach dynamically

    # Replace update_weights to use optimizer
    def update_with_optimizer():
        model.optimizer.update(model.layers)

    model.update_weights = update_with_optimizer

    # Train
    model.train(X_train, y_train,
                X_test, y_test,
                epochs=args.epochs,
                batch_size=args.batch_size)

    # Evaluate
    from sklearn.metrics import f1_score

    logits = model.forward(X_test)
    probs = Softmax.forward(logits)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    test_accuracy = np.mean(y_pred == y_true)
    test_f1 = f1_score(y_true, y_pred, average="macro")

    os.makedirs("src", exist_ok=True)

    # Load previous best score
    if os.path.exists(BEST_SCORE_PATH):
        with open(BEST_SCORE_PATH, "r") as f:
            best_score = float(f.read())
    else:
        best_score = -1

    # Save only if better
    if test_f1 > best_score:
        print("New Best Model Found! Saving...")

        best_weights = model.get_weights()
        np.save(BEST_MODEL_PATH, best_weights)

        with open(BEST_SCORE_PATH, "w") as f:
            f.write(str(test_f1))

        with open(BEST_CONFIG_PATH, "w") as f:
            json.dump(vars(args), f, indent=4)

        print(f"Best model saved with F1: {test_f1:.4f}")
    else:
        print("Model not better than current best.")

    wandb.log({
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    })

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1: {test_f1:.4f}")


if __name__ == "__main__":
        main()