"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
from .neural_layer import LinearLayer
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .objective_functions import CrossEntropy, MeanSquaredError


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments (argparse.Namespace) for configuring the network.
        """
        self.num_layers = cli_args.num_layers

        # Support both a list of sizes and a single integer for hidden_size
        if isinstance(cli_args.hidden_size, list):
            assert len(cli_args.hidden_size) == cli_args.num_layers, (
                "Length of hidden_size list must equal num_layers"
            )
            self.hidden_sizes = cli_args.hidden_size
        else:
            self.hidden_sizes = [cli_args.hidden_size] * cli_args.num_layers

        self.activation_name = cli_args.activation
        self.loss_name = cli_args.loss
        self.learning_rate = cli_args.learning_rate
        self.weight_init = cli_args.weight_init

        # Safety constraint per assignment instructions
        assert all(h <= 128 for h in self.hidden_sizes), (
            "Hidden layer size must not exceed 128 neurons"
        )

        # ---- Activation selection ----
        if self.activation_name == "relu":
            self.activation = ReLU
        elif self.activation_name == "sigmoid":
            self.activation = Sigmoid
        elif self.activation_name == "tanh":
            self.activation = Tanh
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        # ---- Loss selection ----
        if self.loss_name == "cross_entropy":
            self.loss_fn = CrossEntropy
        elif self.loss_name == "mse":
            self.loss_fn = MeanSquaredError
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

        # ---- Build layers ----
        self.layers = []
        input_dim = 784   # MNIST flattened input size
        output_dim = 10   # Number of classes

        for hidden_dim in self.hidden_sizes:
            self.layers.append(
                LinearLayer(input_dim, hidden_dim, weight_init=self.weight_init)
            )
            input_dim = hidden_dim

        # Output layer (no activation — returns raw logits)
        self.layers.append(
            LinearLayer(input_dim, output_dim, weight_init=self.weight_init)
        )

        # Caches filled during forward pass (used by backward)
        self.Z_cache = []
        self.A_cache = []

    # ------------------------------------------------------------------
    # Weight serialization helpers (required by autograder)
    # ------------------------------------------------------------------

    def get_weights(self):
        """Return a dict of all layer weights keyed as W{i} and b{i}."""
        weights = {}
        for idx, layer in enumerate(self.layers):
            weights[f"W{idx}"] = layer.W
            weights[f"b{idx}"] = layer.b
        return weights

    def set_weights(self, weights):
        """Load weights from a dict produced by get_weights()."""
        for idx, layer in enumerate(self.layers):
            layer.W = weights[f"W{idx}"]
            layer.b = weights[f"b{idx}"]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Forward propagation through all layers.

        Args:
            X: Input array of shape (batch_size, 784).

        Returns:
            logits : Raw linear outputs of the last layer, shape (batch_size, 10).
                     Softmax is NOT applied — the model returns logits as required.

        Softmax probabilities are available via self.probs after each forward call.
        """
        A = X
        self.Z_cache = []
        self.A_cache = [X]

        # Hidden layers: linear + activation
        for layer in self.layers[:-1]:
            Z = layer.forward(A)
            A = self.activation.forward(Z)
            self.Z_cache.append(Z)
            self.A_cache.append(A)

        # Output layer: linear only (no activation — raw logits)
        Z_out = self.layers[-1].forward(A)
        self.Z_cache.append(Z_out)

        # Store probs internally so evaluate/train can use them without re-computing
        self.probs = Softmax.forward(Z_out)

        return Z_out   # <-- logits only (as required by autograder)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute and store gradients in each layer.

        Args:
            y_true : Ground-truth labels. Accepts:
                       - one-hot array  of shape (batch_size, 10), or
                       - integer labels of shape (batch_size,)
            y_pred : Raw logits of shape (batch_size, 10) from forward().

        Returns:
            grad_W_list : List of weight gradients ordered from first → last layer.
            grad_b_list : List of bias   gradients ordered from first → last layer.

        Gradients are also stored on each layer as layer.grad_W and layer.grad_b
        for the autograder to inspect.
        """
        # Convert integer labels to one-hot if necessary
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_labels = y_true.ravel().astype(int)
            y_true = np.eye(y_pred.shape[1])[y_labels]

        # Gradient of loss w.r.t. logits (accounts for softmax + loss together)
        dZ = self.loss_fn.derivative(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        # ---- OUTPUT LAYER ----
        dA = self.layers[-1].backward(dZ)
        grad_W_list.append(self.layers[-1].grad_W)
        grad_b_list.append(self.layers[-1].grad_b)

        # ---- HIDDEN LAYERS (reversed: last hidden → first hidden) ----
        for i in reversed(range(len(self.layers) - 1)):
            Z = self.Z_cache[i]
            dZ = dA * self.activation.derivative(Z)
            dA = self.layers[i].backward(dZ)
            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)

        # Reverse so lists are ordered first layer → last layer
        grad_W_list.reverse()
        grad_b_list.reverse()

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list

        return self.grad_W, self.grad_b

    # ------------------------------------------------------------------
    # Weight update (plain SGD; overridden in train.py by optimizer)
    # ------------------------------------------------------------------

    def update_weights(self):
        """Update weights using plain SGD (overridden when an optimizer is attached)."""
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.grad_W
            layer.b -= self.learning_rate * layer.grad_b

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        """
        Train the network for the given number of epochs with mini-batches.

        Args:
            X_train, y_train : Training data (y_train should be one-hot).
            X_test,  y_test  : Validation / test data.
            epochs            : Number of full passes over training data.
            batch_size        : Mini-batch size.
        """
        n = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(n)
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0
            num_batches = 0

            # Track last-batch grad info for epoch-level logging
            last_grad_norm = 0.0
            last_grad_neurons = [0.0] * 5

            for i in range(0, n, batch_size):
                X_batch = X_train[i: i + batch_size]
                y_batch = y_train[i: i + batch_size]

                # Forward
                logits = self.forward(X_batch)

                # Loss
                loss = self.loss_fn.forward(y_batch, logits)
                epoch_loss += loss
                num_batches += 1

                # Backward
                self.backward(y_batch, logits)

                # Cache gradient info from last batch (logged once per epoch below)
                last_grad_norm = float(np.linalg.norm(self.layers[0].grad_W))
                for neuron_idx in range(min(5, self.layers[0].grad_W.shape[1])):
                    last_grad_neurons[neuron_idx] = float(
                        np.linalg.norm(self.layers[0].grad_W[:, neuron_idx])
                    )

                # Weight update
                self.update_weights()

            epoch_loss /= num_batches

            train_accuracy = self.evaluate(X_train, y_train)
            test_accuracy  = self.evaluate(X_test,  y_test)

            # ---- Single wandb.log per epoch (fast) ----
            log_dict = {
                "epoch":          epoch + 1,
                "train_loss":     epoch_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy":  test_accuracy,
                "grad_norm_layer1": last_grad_norm,
            }
            # Activation histogram from last batch of this epoch
            try:
                if len(self.A_cache) > 1:
                    log_dict["activation_hist_layer1"] = wandb.Histogram(self.A_cache[1])
                for neuron_idx, val in enumerate(last_grad_neurons):
                    log_dict[f"grad_neuron_{neuron_idx}"] = val
                wandb.log(log_dict)
            except Exception:
                pass  # wandb not available (e.g. autograder environment)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X, y):
        """
        Compute accuracy on a dataset.

        Args:
            X : Input features.
            y : One-hot labels of shape (n_samples, 10).

        Returns:
            Accuracy as a float in [0, 1].
        """
        self.forward(X)
        predictions = np.argmax(self.probs, axis=1)
        true_labels = np.argmax(y, axis=1)
        return float(np.mean(predictions == true_labels))


# ------------------------------------------------------------------
# Utility: numerical gradient check
# ------------------------------------------------------------------

def gradient_check(model, X, y, epsilon=1e-6):
    """
    Verify analytical gradients against numerical gradients for the first layer.

    A difference < 1e-5 indicates the backward pass is correct.
    """
    logits = model.forward(X)
    model.backward(y, logits)

    layer = model.layers[0]
    analytical_grad = layer.grad_W.copy()
    numerical_grad  = np.zeros_like(layer.W)

    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            orig = layer.W[i, j]

            layer.W[i, j] = orig + epsilon
            plus_logits = model.forward(X)
            plus_loss = model.loss_fn.forward(y, plus_logits)

            layer.W[i, j] = orig - epsilon
            minus_logits = model.forward(X)
            minus_loss = model.loss_fn.forward(y, minus_logits)

            numerical_grad[i, j] = (plus_loss - minus_loss) / (2 * epsilon)
            layer.W[i, j] = orig   # restore

    diff = np.linalg.norm(analytical_grad - numerical_grad)
    print(f"Gradient Check Difference: {diff:.2e}")
    return diff