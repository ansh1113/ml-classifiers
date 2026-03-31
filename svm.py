"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes (C)
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: regularization constant λ (weight decay strength)
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Gradient of the multi-class (Weston-Watkins) SVM hinge loss.

        Loss for one sample (x, y):
            L_i = Σ_{c ≠ y}  max(0,  w_c·x - w_y·x + 1)

        Each wrong class c contributes a term when:
            w_c·x - w_y·x + 1 > 0   (i.e. the margin is violated)

        Gradients w.r.t. the weight rows:
            ∂L_i / ∂w_c  =  +x          if c ≠ y and margin violated
            ∂L_i / ∂w_y  =  -(#violated) · x

        With L2 regularization:
            total_loss = hinge_loss + (λ/2) ‖W‖²
            ∂total / ∂w_c  +=  λ · w_c

        Parameters:
            X_train: (N, D) mini-batch of features
            y_train: (N,)  ground-truth labels

        Returns:
            gradient of shape (C, D), same as self.w
        """
        N, D = X_train.shape
        C = self.n_class

        # 1. Compute all scores: (N, C)
        scores = X_train @ self.w.T

        # 2. correct class score for each sample: (N, 1)
        #    select the score corresponding to y_train[i] for each i
        correct_class_scores = scores[np.arange(N), y_train][:, np.newaxis]

        # 3. Calculate margins: (N, C)
        #    Delta = 1.0. We want w_j*x - w_yi*x + 1 > 0
        margins = np.maximum(0, scores - correct_class_scores + 1.0)
        
        # 4. 
        margins[np.arange(N), y_train] = 0

        # 5. binary mask of violations (where margin > 0)
        #    (N, C) -> 1 if violation, 0 otherwise
        binary_mask = (margins > 0).astype(float)

        # 6. For the correct class y_i, the gradient component is:
        #    - (number of violations) * x_i
        #    count violations per row:
        row_counts = np.sum(binary_mask, axis=1)
        
        #    Subtract this count from the correct class column in the mask
        binary_mask[np.arange(N), y_train] -= row_counts

        # 7. Compute Gradient: (C, N) @ (N, D) = (C, D)
        grad = binary_mask.T @ X_train

        # 8. Average and Regularize
        grad /= N
        grad += self.reg_const * self.w

        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with SGD on mini-batches.

        The SVM loss (soft-margin, multi-class hinge):
            L = (1/N) Σ_i Σ_{c≠y_i} max(0, w_c·x_i − w_{y_i}·x_i + 1)
              + (λ/2) ‖W‖²

        We use Stochastic Gradient Descent:
          1. Shuffle training data each epoch
          2. Split into mini-batches
          3. Compute gradient on each mini-batch
          4. Update: W ← W − lr · ∇L

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
        """Predict by taking the class with the highest linear score.

        Parameters:
            X_test: (N, D) test features

        Returns:
            (N,) predicted class indices
        """
        # (N, D) @ (D, C) = (N, C)
        scores = X_test @ self.w.T
        return np.argmax(scores, axis=1)