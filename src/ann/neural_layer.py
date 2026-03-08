"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features, weight_init='random'):
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight Initialization
        if weight_init == 'random':
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == 'xavier':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        elif weight_init == 'zeros':
            self.W = np.zeros((in_features, out_features))
        else:
            raise ValueError("Unsupported weight initialization")

        self.b = np.zeros((1, out_features))

        # For storing gradients (required by autograder)
        self.grad_W = None
        self.grad_b = None

        # Cache for backward
        self.X = None

    def forward(self, X):
        """
        X shape: (batch_size, in_features)
        Returns:
        Z shape: (batch_size, out_features)
        """
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        """
        dZ shape: (batch_size, out_features)
        Returns:
        dX shape: (batch_size, in_features)
        """
        m = self.X.shape[0]

        self.grad_W = (self.X.T @ dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m

        dX = dZ @ self.W.T
        return dX