"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import Softmax


class MeanSquaredError:

    @staticmethod
    def forward(y_true, y_pred):
        m = y_true.shape[0]
        return np.sum((y_true - y_pred) ** 2) / m

    @staticmethod
    def derivative(y_true, y_pred):
        m = y_true.shape[0]
        return (2 / m) * (y_pred - y_true)


class CrossEntropy:

    @staticmethod
    def forward(y_true, logits):

        probs = Softmax.forward(logits)

        m = y_true.shape[0]

        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(probs)) / m

        return loss

    @staticmethod
    def derivative(y_true, logits):

        probs = Softmax.forward(logits)

        m = y_true.shape[0]

        return (probs - y_true) / m