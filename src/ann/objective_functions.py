"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np
from .activations import Softmax


def _to_onehot(y, num_classes=10):
    """
    Convert integer class labels to one-hot encoding.

    Accepts:
        y : ndarray of shape (batch,)  — integer labels
            ndarray of shape (batch,1) — integer labels as column vector
            ndarray of shape (batch, num_classes) — already one-hot (returned as-is)

    Returns:
        ndarray of shape (batch, num_classes)
    """
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y                              # already one-hot
    labels = y.ravel().astype(int)
    return np.eye(num_classes)[labels]


class MeanSquaredError:
    """Mean Squared Error loss (with softmax applied to logits first)."""

    @staticmethod
    def forward(y_true, y_pred):
        """
        Args:
            y_true : One-hot or integer labels.
            y_pred : Raw logits, shape (batch, num_classes).
        Returns:
            Scalar loss value.
        """
        num_classes = y_pred.shape[1]
        y_true = _to_onehot(y_true, num_classes)
        probs = Softmax.forward(y_pred)
        m = y_true.shape[0]
        return float(np.sum((y_true - probs) ** 2) / m)

    @staticmethod
    def derivative(y_true, y_pred):
        """
        Gradient of MSE loss w.r.t. logits.

        Uses the chain rule through the softmax layer.

        Returns:
            dL/dlogits, shape (batch, num_classes).
        """
        num_classes = y_pred.shape[1]
        y_true = _to_onehot(y_true, num_classes)
        probs = Softmax.forward(y_pred)
        m = y_true.shape[0]

        # dL/d(probs)
        dL_dprobs = (2.0 / m) * (probs - y_true)   # (batch, C)

        # Chain through softmax: dL/dz_k = sum_j (dL/dp_j * dp_j/dz_k)
        # dp_j/dz_k = p_k*(delta_jk - p_j)
        # => dL/dz_k = p_k * (dL/dp_k - sum_j(dL/dp_j * p_j))
        dot = np.sum(dL_dprobs * probs, axis=1, keepdims=True)   # (batch, 1)
        dZ = probs * (dL_dprobs - dot)                            # (batch, C)
        return dZ


class CrossEntropy:
    """Categorical Cross-Entropy loss (with softmax applied to logits first)."""

    @staticmethod
    def forward(y_true, logits):
        """
        Args:
            y_true  : One-hot or integer labels.
            logits  : Raw logits, shape (batch, num_classes).
        Returns:
            Scalar loss value.
        """
        num_classes = logits.shape[1]
        y_true = _to_onehot(y_true, num_classes)
        probs = Softmax.forward(logits)
        m = y_true.shape[0]
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1.0 - epsilon)
        loss = -np.sum(y_true * np.log(probs)) / m
        return float(loss)

    @staticmethod
    def derivative(y_true, logits):
        """
        Gradient of Cross-Entropy loss w.r.t. logits.

        Because we use softmax + cross-entropy together, the combined
        gradient simplifies to (probs - y_true) / m.

        Args:
            y_true  : One-hot or integer labels.
            logits  : Raw logits, shape (batch, num_classes).
        Returns:
            dL/dlogits, shape (batch, num_classes).
        """
        num_classes = logits.shape[1]
        y_true = _to_onehot(y_true, num_classes)
        probs = Softmax.forward(logits)
        m = y_true.shape[0]
        return (probs - y_true) / m