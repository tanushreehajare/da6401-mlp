"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import wandb
from ann.activations import Softmax
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ann.neural_network import NeuralNetwork

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """

    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--batch_size", type=int, required=True)

    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--hidden_size", nargs="+", type=int, required=True)

    parser.add_argument("--activation", type=str, required=True,
                        choices=["relu", "sigmoid", "tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])

    parser.add_argument("--weight_init", type=str, choices=["xavier","random"])

    parser.add_argument("--learning_rate", type=float, default=0.001)

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

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )

    y_test_onehot = np.eye(10)[y_test]

    return X_test, y_test, y_test_onehot

def load_model(model_path, args):

    # Recreate model architecture
    model = NeuralNetwork(args)

    # Load saved weights
    weights = np.load(model_path, allow_pickle=True).item()

    for idx, layer in enumerate(model.layers):
        layer.W = weights[f"W{idx}"]
        layer.b = weights[f"b{idx}"]

    return model


def evaluate_model(model, X_test, y_test, y_test_onehot):
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """

    logits = model.forward(X_test)
    probs = Softmax.forward(logits)
    y_pred = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    import wandb

    cm = wandb.plot.confusion_matrix(
        y_true=y_test,
        preds=y_pred,
        class_names=[str(i) for i in range(10)]
    )

    wandb.log({"confusion_matrix": cm})

    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Log 10 misclassified images
    mis_idx = np.where(y_pred != y_test)[0][:10]
    table = wandb.Table(columns=["image", "true", "pred"])

    for idx in mis_idx:
        img = X_test[idx].reshape(28,28)
        table.add_data(wandb.Image(img), int(y_test[idx]), int(y_pred[idx]))

    wandb.log({"misclassified_examples": table})

    loss = model.loss_fn.forward(y_test_onehot, logits)

    results = {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    wandb.log({
        "test_loss": loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })

    return results


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    wandb.init(project="sweeps-mlp-last")

    X_test, y_test, y_test_onehot = load_data(args.dataset)

    model = load_model(args.model_path, args)

    results = evaluate_model(model, X_test, y_test, y_test_onehot)

    print("\n===== Evaluation Results =====")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")

    return results


if __name__ == '__main__':
    main()
