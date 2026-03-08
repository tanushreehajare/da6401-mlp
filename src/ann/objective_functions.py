"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import Softmax

class MeanSquaredError:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        m = y_true.shape[0]
        return 2 * (y_pred - y_true)


class CrossEntropy:

    @staticmethod
    def forward(y_true, logits):
        probs = Softmax.forward(logits)
        m = y_true.shape[0]
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(probs)) / m

    @staticmethod
    def derivative(y_true, logits):
        probs = Softmax.forward(logits)
        m = y_true.shape[0]
        return (probs - y_true) 