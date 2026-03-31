"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
            threshold: decision boundary (default 0.5)
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function: maps any real number to (0, 1).

        σ(z) = 1 / (1 + e^{-z})

        Numerically stable version:
        - For z >= 0: use the standard formula directly
        - For z <  0: rewrite as e^z / (1 + e^z) to avoid overflow
            from exp(-z) when z is very negative.

        Parameters:
            z: input array of any shape

        Returns:
            element-wise sigmoid values in (0, 1)
        """
        result = np.zeros_like(z, dtype=float)

        # Positive entries: standard formula is safe
        pos = z >= 0
        result[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

        # Negative entries: equivalent form avoids exp of large positive number
        neg = ~pos
        exp_z = np.exp(z[neg])
        result[neg] = exp_z / (1.0 + exp_z)

        return result

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with gradient descent on binary cross-entropy loss.

        The logistic regression model computes:
            p(y=1 | x) = σ(w · x)

        Loss for one sample: -[y log(p) + (1-y) log(1-p)]

        Gradient of loss w.r.t. w:
            ∇L = (p - y) · x

        So the weight update (gradient descent) is:
            w ← w - lr · (p - y) · x

        In matrix form for a whole batch of N samples:
            w ← w - (lr / N) · Xᵀ · (p - y)

        NOTE: labels here are 0/1 (not -1/+1), so the formula above
        applies directly — no label remapping needed.

        Parameters:
            X_train: (N, D) training features
            y_train: (N,)  training labels in {0, 1}
        """
        N, D = X_train.shape

        self.w = np.random.uniform(-1.0, 1.0, D) * 0.01

        for epoch in range(self.epochs):
            # scores shape: (N,)
            scores = X_train @ self.w          
            # p_hat shape: (N,)  — each entry is P(y=1 | x_i)
            p_hat = self.sigmoid(scores)

            # Error / residual: positive means over-predicted, negative means under
            error = p_hat - y_train            # shape (N,)

            # Gradient: Xᵀ · error  averaged over the batch
            grad = X_train.T @ error / N      # shape (D,)

            # Gradient descent step
            self.w -= self.lr * grad

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict binary labels using learned weights.

        Computes p = σ(w · x) for each test point.
        Predicts label 1 if p >= threshold, else 0.

        Parameters:
            X_test: (N, D) test features

        Returns:
            (N,) array of predicted labels in {0, 1}
        """
        scores = X_test @ self.w
        p_hat = self.sigmoid(scores)
        # decision threshold
        return (p_hat >= self.threshold).astype(int)