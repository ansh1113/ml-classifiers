"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes (C)
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the multi-class perceptron.

        The multi-class perceptron keeps one weight vector w_c per class.
        Stored in a matrix self.w of shape (C, D).

        Prediction rule:
            ŷ = argmax_c  w_c · x

        Update rule (only fires on a misclassification):
            For each sample (x, y):
              ŷ = argmax_c (w_c · x)
              if ŷ ≠ y:
                  w_y  ← w_y  + lr · x    # reward the correct class
                  w_ŷ  ← w_ŷ  - lr · x   # penalise the wrongly predicted class

        Intuition: we're nudging the correct class's hyperplane to score
        higher on x, and nudging the wrong class's hyperplane to score lower.

        Parameters:
            X_train: (N, D) training features
            y_train: (N,)  training labels in {0, …, C-1}
        """
        N, D = X_train.shape
        C = self.n_class

        self.w = np.zeros((C, D))
        
        decay_rate = 0.5

        for epoch in range(self.epochs):
            current_lr = self.lr * np.exp(-epoch / decay_rate)

            perm = np.random.permutation(N)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            for i in range(N):
                x = X_shuffled[i]       
                y = y_shuffled[i]       

                scores = self.w @ x  
                y_hat = np.argmax(scores)

                if y_hat != y:
                    self.w[y]     += current_lr * x   
                    self.w[y_hat] -= current_lr * x

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict class labels by taking the argmax score.

        scores = X_test @ w.T   →  shape (N, C)
        ŷ_i = argmax over columns

        Parameters:
            X_test: (N, D) test features

        Returns:
            (N,) predicted class indices
        """
        # (N, D) @ (D, C) = (N, C)  — score matrix
        scores = X_test @ self.w.T
        return np.argmax(scores, axis=1)