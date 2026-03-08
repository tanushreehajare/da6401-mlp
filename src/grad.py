import numpy as np
import argparse
from ann.neural_network import NeuralNetwork, gradient_check

# dummy config
config = argparse.Namespace(
    num_layers=2,
    hidden_size=[64,64],
    activation="relu",
    loss="cross_entropy",
    learning_rate=0.01,
    weight_init="xavier"
)

model = NeuralNetwork(config)

# small random dataset
X = np.random.randn(5,784)
y_labels = np.random.randint(0,10,size=5)
y = np.eye(10)[y_labels]

# run gradient check
gradient_check(model, X, y)