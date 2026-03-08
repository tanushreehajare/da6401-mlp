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
            cli_args: Command-line arguments for configuring the network
        """
        self.num_layers = cli_args.num_layers
        if isinstance(cli_args.hidden_size, list):
            assert len(cli_args.hidden_size) == cli_args.num_layers
            self.hidden_sizes = cli_args.hidden_size
        else:
            self.hidden_sizes = [cli_args.hidden_size] * cli_args.num_layers
        self.activation_name = cli_args.activation
        self.loss_name = cli_args.loss
        self.learning_rate = cli_args.learning_rate
        self.weight_init = cli_args.weight_init

        # Safety constraint (assignment instruction)
        assert all(h <= 128 for h in self.hidden_sizes)

        # Activation selection
        if self.activation_name == "relu":
            self.activation = ReLU
        elif self.activation_name == "sigmoid":
            self.activation = Sigmoid
        elif self.activation_name == "tanh":
            self.activation = Tanh
        else:
            raise ValueError("Unsupported activation")

        # Loss selection
        if self.loss_name == "cross_entropy":
            self.loss_fn = CrossEntropy
        elif self.loss_name == "mse":
            self.loss_fn = MeanSquaredError
        else:
            raise ValueError("Unsupported loss")

        self.layers = []

        input_dim = 784  # MNIST flattened size
        output_dim = 10  # number of classes

        # Hidden layers
        for hidden_dim in self.hidden_sizes:
            self.layers.append(
                LinearLayer(input_dim, hidden_dim, weight_init=self.weight_init)
            )
            input_dim = hidden_dim

        # Output layer
        self.layers.append(
            LinearLayer(input_dim, output_dim, weight_init=self.weight_init)
        )

    def get_weights(self):
        weights = {}
        for idx, layer in enumerate(self.layers):
            weights[f"W{idx}"] = layer.W
            weights[f"b{idx}"] = layer.b
        return weights

    def set_weights(self, weights):
        for idx, layer in enumerate(self.layers):
            layer.W = weights[f"W{idx}"]
            layer.b = weights[f"b{idx}"]
    
    def forward(self, X):
        """
        Forward propagation through all layers.
        
        Args:
            X: Input data
            
        Returns:
            Output logits
        """

        A = X
        self.Z_cache = []
        self.A_cache = [X]

        for layer in self.layers[:-1]:
            Z = layer.forward(A)
            A = self.activation.forward(Z)
            self.Z_cache.append(Z)
            self.A_cache.append(A)

        Z_out = self.layers[-1].forward(A)
        self.Z_cache.append(Z_out)

        return Z_out
    
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        
        Args:
            y_true: True labels
            y_pred: Predicted outputs
            
        Returns:
            return grad_w, grad_b
        """

        # gradient of loss wrt logits
        dZ = self.loss_fn.derivative(y_true, y_pred)

        grad_W_list = []
        grad_b_list = []

        # ---- OUTPUT LAYER ----
        dA = self.layers[-1].backward(dZ)

        grad_W_list.append(self.layers[-1].grad_W)
        grad_b_list.append(self.layers[-1].grad_b)

        # ---- HIDDEN LAYERS ----
        for i in reversed(range(len(self.layers)-1)):

            Z = self.Z_cache[i]

            # activation derivative
            dZ = dA * self.activation.derivative(Z)

            dA = self.layers[i].backward(dZ)

            grad_W_list.append(self.layers[i].grad_W)
            grad_b_list.append(self.layers[i].grad_b)

        # store gradients as object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i,(gw,gb) in enumerate(zip(grad_W_list,grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b
    
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.grad_W
            layer.b -= self.learning_rate * layer.grad_b
    
    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size):
        """
        Train the network for specified epochs.
        """

        n = X_train.shape[0]

        for epoch in range(epochs):

            permutation = np.random.permutation(n)
            X_train = X_train[permutation]
            y_train = y_train[permutation]

            epoch_loss = 0

            for i in range(0, n, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)
                y_pred = Softmax.forward(logits)

                # Q2.5 Activation histogram
                if epoch == 0 and i < 50 and len(self.A_cache) > 1:
                    wandb.log({
                        "activation_hist_layer1":
                        wandb.Histogram(self.A_cache[1])
                    })

                loss = self.loss_fn.forward(y_batch, y_pred)
                epoch_loss += loss

                self.backward(y_batch, y_pred)

                grad_norm = np.linalg.norm(self.layers[0].grad_W)
                wandb.log({"grad_norm_layer1": grad_norm})

                # Log gradients of first 5 neurons
                for neuron_idx in range(min(5, self.layers[0].grad_W.shape[1])):
                    wandb.log({
                        f"grad_neuron_{neuron_idx}":
                        np.linalg.norm(self.layers[0].grad_W[:, neuron_idx])
                    })

                self.update_weights()

            epoch_loss /= (n / batch_size)

            train_accuracy = self.evaluate(X_train, y_train)
            test_accuracy = self.evaluate(X_test, y_test)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy
            })
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """

        logits = self.forward(X)
        probs = Softmax.forward(logits)
        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y, axis=1)

        accuracy = np.mean(predictions == true_labels)

        return accuracy
    
def gradient_check(model, X, y, epsilon=1e-6):
    """
    Numerical gradient checking for first layer weights.
    """
    # Forward
    logits = model.forward(X)
    y_pred = Softmax.forward(logits)
    model.backward(y, y_pred)

    layer = model.layers[0]  # check first layer
    analytical_grad = layer.grad_W.copy()

    numerical_grad = np.zeros_like(layer.W)

    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            
            original_value = layer.W[i, j]

            # f(w + eps)
            layer.W[i, j] = original_value + epsilon
            plus_logits = model.forward(X)
            plus_loss = model.loss_fn.forward(y, Softmax.forward(plus_logits))

            # f(w - eps)
            layer.W[i, j] = original_value - epsilon
            minus_logits = model.forward(X)
            minus_loss = model.loss_fn.forward(y, Softmax.forward(minus_logits))

            numerical_grad[i, j] = (plus_loss - minus_loss) / (2 * epsilon)

            layer.W[i, j] = original_value  # restore

    difference = np.linalg.norm(analytical_grad - numerical_grad)
    print("Gradient Check Difference:", difference)