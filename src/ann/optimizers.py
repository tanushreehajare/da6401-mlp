"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np
    
class SGD:
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def update(self, layers):
        for layer in layers:
            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b
            
class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):

            if idx not in self.velocities:
                self.velocities[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            vW = self.velocities[idx]["vW"]
            vb = self.velocities[idx]["vb"]

            vW = self.beta * vW + (1 - self.beta) * layer.grad_W
            vb = self.beta * vb + (1 - self.beta) * layer.grad_b

            layer.W -= self.lr * vW
            layer.b -= self.lr * vb

            self.velocities[idx]["vW"] = vW
            self.velocities[idx]["vb"] = vb
            
class NAG:
    def __init__(self, learning_rate=0.01, beta=0.9, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.weight_decay = weight_decay
        self.velocities = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):

            if idx not in self.velocities:
                self.velocities[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            vW_prev = self.velocities[idx]["vW"]
            vb_prev = self.velocities[idx]["vb"]

            vW = self.beta * vW_prev + layer.grad_W
            vb = self.beta * vb_prev + layer.grad_b

            layer.W -= self.lr * (self.beta * vW + layer.grad_W)
            layer.b -= self.lr * (self.beta * vb + layer.grad_b)

            self.velocities[idx]["vW"] = vW
            self.velocities[idx]["vb"] = vb
            
class RMSProp:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon
        self.weight_decay = weight_decay
        self.squares = {}

    def update(self, layers):
        for idx, layer in enumerate(layers):

            if idx not in self.squares:
                self.squares[idx] = {
                    "sW": np.zeros_like(layer.W),
                    "sb": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            sW = self.squares[idx]["sW"]
            sb = self.squares[idx]["sb"]

            sW = self.beta * sW + (1 - self.beta) * (layer.grad_W ** 2)
            sb = self.beta * sb + (1 - self.beta) * (layer.grad_b ** 2)

            layer.W -= self.lr * layer.grad_W / (np.sqrt(sW) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(sb) + self.eps)

            self.squares[idx]["sW"] = sW
            self.squares[idx]["sb"] = sb
        