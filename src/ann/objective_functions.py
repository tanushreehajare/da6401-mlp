"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from .activations import Softmax

def _as_probabilities(values):
    """Accept either logits or already-normalized probabilities."""
    if values.ndim == 2:
        row_sums = np.sum(values, axis=1)
        looks_like_probs = (
            np.all(values >= 0)
            and np.all(values <= 1)
            and np.allclose(row_sums, 1.0, atol=1e-6)
        )
        if looks_like_probs:
            return values
    return Softmax.forward(values)

class MeanSquaredError:
    @staticmethod
    def forward(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        n_elements = np.prod(y_true.shape)
        return (2 / n_elements) * (y_pred - y_true)


class CrossEntropy:

    @staticmethod
    def forward(y_true, logits):
        probs = _as_probabilities(logits)
        m = y_true.shape[0]
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(probs)) / m

    @staticmethod
    def derivative(y_true, logits):
        probs = _as_probabilities(logits)
        m = y_true.shape[0]
        return (probs - y_true) / m