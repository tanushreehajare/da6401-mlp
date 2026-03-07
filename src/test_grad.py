import numpy as np
from ann.neural_network import NeuralNetwork, gradient_check

np.random.seed(42)

class Args:
    num_hidden_layers = 2
    hidden_size = 64
    activation = "relu"
    loss = "cross_entropy"
    learning_rate = 0.001
    weight_init = "xavier"

args = Args()
model = NeuralNetwork(args)

X = np.random.randn(5, 784)

y = np.zeros((5,10))
for i in range(5):
    y[i, np.random.randint(0,10)] = 1

gradient_check(model, X, y)