import numpy as np
import argparse

from ann.neural_network import NeuralNetwork

best_config = argparse.Namespace(
    dataset="mnist",
    epochs=2,
    batch_size=64,
    loss="cross_entropy",
    optimizer="sgd",
    weight_decay=0.0,
    learning_rate=0.01,
    num_layers=2,
    hidden_size=128,
    activation="relu",
    weight_init="xavier"
)

model = NeuralNetwork(best_config)

weights = np.load("src/best_model.npy", allow_pickle=True).item()

for i, layer in enumerate(model.layers):
    layer.W = weights[f"W{i}"]
    layer.b = weights[f"b{i}"]

print("Model loaded successfully")