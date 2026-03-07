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
            
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.0):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layers):

        self.t += 1

        for idx, layer in enumerate(layers):

            if idx not in self.m:
                self.m[idx] = {
                    "mW": np.zeros_like(layer.W),
                    "mb": np.zeros_like(layer.b)
                }
                self.v[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            mW = self.m[idx]["mW"]
            mb = self.m[idx]["mb"]
            vW = self.v[idx]["vW"]
            vb = self.v[idx]["vb"]

            mW = self.beta1 * mW + (1 - self.beta1) * layer.grad_W
            mb = self.beta1 * mb + (1 - self.beta1) * layer.grad_b

            vW = self.beta2 * vW + (1 - self.beta2) * (layer.grad_W ** 2)
            vb = self.beta2 * vb + (1 - self.beta2) * (layer.grad_b ** 2)

            mW_hat = mW / (1 - self.beta1 ** self.t)
            mb_hat = mb / (1 - self.beta1 ** self.t)
            vW_hat = vW / (1 - self.beta2 ** self.t)
            vb_hat = vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

            self.m[idx]["mW"] = mW
            self.m[idx]["mb"] = mb
            self.v[idx]["vW"] = vW
            self.v[idx]["vb"] = vb
   
class Nadam(Adam):

    def update(self, layers):

        self.t += 1

        for idx, layer in enumerate(layers):

            if idx not in self.m:
                self.m[idx] = {
                    "mW": np.zeros_like(layer.W),
                    "mb": np.zeros_like(layer.b)
                }
                self.v[idx] = {
                    "vW": np.zeros_like(layer.W),
                    "vb": np.zeros_like(layer.b)
                }

            if self.weight_decay > 0:
                layer.grad_W += self.weight_decay * layer.W

            mW = self.m[idx]["mW"]
            mb = self.m[idx]["mb"]
            vW = self.v[idx]["vW"]
            vb = self.v[idx]["vb"]

            mW = self.beta1 * mW + (1 - self.beta1) * layer.grad_W
            mb = self.beta1 * mb + (1 - self.beta1) * layer.grad_b

            vW = self.beta2 * vW + (1 - self.beta2) * (layer.grad_W ** 2)
            vb = self.beta2 * vb + (1 - self.beta2) * (layer.grad_b ** 2)

            mW_hat = mW / (1 - self.beta1 ** self.t)
            mb_hat = mb / (1 - self.beta1 ** self.t)
            vW_hat = vW / (1 - self.beta2 ** self.t)
            vb_hat = vb / (1 - self.beta2 ** self.t)

            nesterov_W = self.beta1 * mW_hat + (1 - self.beta1) * layer.grad_W
            nesterov_b = self.beta1 * mb_hat + (1 - self.beta1) * layer.grad_b

            layer.W -= self.lr * nesterov_W / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * nesterov_b / (np.sqrt(vb_hat) + self.eps)

            self.m[idx]["mW"] = mW
            self.m[idx]["mb"] = mb
            self.v[idx]["vW"] = vW
            self.v[idx]["vb"] = vb         
