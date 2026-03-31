"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes (C)
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: L2 regularization constant λ
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Gradient of cross-entropy loss + L2 regularization.

        Forward pass:
            logits  = X @ W.T          shape (N, C)   — raw scores
            shifted = logits - max     shape (N, C)   — numeric stability trick
            softmax = exp(shifted) / Σ exp(shifted)   shape (N, C)

        Loss (cross-entropy):
            L = -(1/N) Σ_i log( softmax[i, y_i] )

        The beautiful gradient:
            ∂L/∂W_c = (1/N) Σ_i  (p_{i,c} - 1{y_i = c}) · x_i

        In matrix form:
            d_scores = softmax                    (N, C)
            d_scores[i, y_i] -= 1  for all i
            ∂L/∂W = d_scores.T @ X / N           (C, D)

        With regularization:
            total_grad = data_grad + λ · W

        Parameters:
            X_train: (N, D) mini-batch
            y_train: (N,)  labels

        Returns:
            gradient of shape (C, D)
        """
        N = X_train.shape[0]

        # Raw class scores: (N, C)
        logits = X_train @ self.w.T

        # Subtract row-wise max for numerical stability (doesn't change softmax output)
        logits -= logits.max(axis=1, keepdims=True)

        # Softmax probabilities: (N, C)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # d_scores: starts as probability matrix, then subtract 1 at correct class
        d_scores = probs.copy()                     # (N, C)
        d_scores[np.arange(N), y_train] -= 1.0      # ∂L/∂logit for correct class

        # Chain rule to get gradient w.r.t. W
        grad = d_scores.T @ X_train / N             # (C, D)

        # L2 regularization gradient
        grad += self.reg_const * self.w

        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with SGD on mini-batches.

        Softmax + cross-entropy loss:
            L = -(1/N) Σ_i log( exp(w_{y_i}·x_i) / Σ_c exp(w_c·x_i) )
              + (λ/2) ‖W‖²

        Mini-batch SGD:
          1. Shuffle data each epoch
          2. Take mini-batch of size B
          3. Compute gradient on batch
          4. W ← W − lr · grad

        Parameters:
            X_train: (N, D) training features
            y_train: (N,)  training labels
        """
        N, D = X_train.shape
        C = self.n_class

        self.w = np.zeros((C, D))
        
        batch_size = 128
        decay_rate = 20.0 

        for epoch in range(self.epochs):
            current_lr = self.lr * np.exp(-epoch / decay_rate)
            
            perm = np.random.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                grad = self.calc_gradient(X_shuffled[start:end], y_shuffled[start:end])
                self.w -= current_lr * grad

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict the class with the highest softmax score.

        Parameters:
            X_test: (N, D) test features

        Returns:
            (N,) predicted class indices
        """
        logits = X_test @ self.w.T       # (N, C)
        return np.argmax(logits, axis=1)