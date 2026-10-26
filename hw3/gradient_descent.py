from typing import Tuple

import numpy as np
from encoder import OneHotEncoder


class GradientDescent:
    """
    Logistic regression classifier trained with batch gradient descent.

    Attributes:
        learning_rate: Step size used for each gradient descent update.
        n_iters: Number of gradient descent iterations.
        weights: Weight vector, one entry per encoded feature (None until fit).
        bias: Bias (intercept) term (None until fit).
        losses: Binary cross-entropy loss recorded at every iteration.
        encoder: Encoder used to encode the data.
    """

    def __init__(self, learning_rate: float = 0.1, n_iters: int = 100, encoder: str = "onehot") -> None:
        """
        Constructor for GradientDescent classifier. Stores the hyperparameters,
        initializes the model parameters to None, creates an empty list to track
        the loss, and initializes the encoder.

        Parameters:
            learning_rate (float): Step size for each update.
            n_iters (int): Number of gradient descent iterations.
            encoder (str): Name of the encoder to use.

        Returns:
            None

        Example:
            >>> gd = GradientDescent(learning_rate=0.5, n_iters=200)
            >>> gd.learning_rate
            0.5
            >>> gd.n_iters
            200
            >>> gd.weights
            >>> gd.bias
            >>> gd.losses
            []
            >>> gd.encoder
            OneHotEncoder()
        """

        # >>> YOUR CODE HERE >>>
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
        # <<< END OF YOUR CODE <<<


        if encoder == "onehot":
            # >>> YOUR CODE HERE >>>
            self.encoder = OneHotEncoder()
            # <<< END OF YOUR CODE <<<
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid function 1 / (1 + exp(-z)) element-wise. Clip z to
        [-500, 500] first so that np.exp does not overflow.

        Parameters:
            z (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid of z, with values in (0, 1).

        Example:
            >>> GradientDescent._sigmoid(np.array([0.0, 2.0, -2.0]))
            array([0.5       , 0.88079708, 0.11920292])
            >>> GradientDescent._sigmoid(np.array([-1000.0, 1000.0]))
            array([7.12457641e-218, 1.00000000e+000])
        """

        # >>> YOUR CODE HERE >>>
        z_clipped = np.clip(z, -500, 500)
        sigmoid = 1 / (1 + np.exp(-z_clipped))
        # <<< END OF YOUR CODE <<<
        return sigmoid

    def _compute_loss(self, y: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Compute the average binary cross-entropy loss:

            L = -mean( y * log(p) + (1 - y) * log(1 - p) )

        Clip p to [1e-15, 1 - 1e-15] first so that np.log never sees 0.

        Parameters:
            y (np.ndarray): True labels (0 or 1).
            y_prob (np.ndarray): Predicted probabilities that y = 1.

        Returns:
            float: Average binary cross-entropy loss.

        Example:
            >>> gd = GradientDescent()
            >>> y = np.array([1, 0, 1, 0])
            >>> y_prob = np.array([0.9, 0.2, 0.6, 0.4])
            >>> round(gd._compute_loss(y, y_prob), 6)
            0.337539
        """

        # >>> YOUR CODE HERE >>>
        p = np.clip(y_prob, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        # <<< END OF YOUR CODE <<<
        return float(loss)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the gradient of the binary cross-entropy loss with respect to
        the weights and the bias:

            dL/dw = X^T (p - y) / n
            dL/db = mean(p - y)

        Parameters:
            X (np.ndarray): Encoded features, shape (n, d).
            y (np.ndarray): True labels, shape (n,).
            y_prob (np.ndarray): Predicted probabilities, shape (n,).

        Returns:
            Tuple[np.ndarray, float]: dw with shape (d,), db.

        Example:
            >>> gd = GradientDescent()
            >>> X = np.array([[1, 0], [0, 1], [1, 1]])
            >>> y = np.array([1, 0, 1])
            >>> y_prob = np.array([0.5, 0.5, 0.5])
            >>> dw, db = gd._compute_gradients(X, y, y_prob)
            >>> dw
            array([-0.33333333,  0.        ])
            >>> round(db, 6)
            -0.166667
        """

        # >>> YOUR CODE HERE >>>
        error = y_prob - y
        dw = X.T @ error/len(y)
        db = np.mean(error)
        # <<< END OF YOUR CODE <<<
        return dw, float(db)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier with batch gradient descent. Start from all-zero
        weights and a zero bias, then for each of the n_iters iterations:
        compute the predicted probabilities, record the loss, compute the
        gradients, and step the weights and bias against the gradient.

        Parameters:
            X (np.ndarray): Training examples.
            y (np.ndarray): Training labels.

        Returns:
            None

        Example:
            >>> gd = GradientDescent(learning_rate=0.5, n_iters=100)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> gd.fit(X, y)
            >>> gd.weights
            array([ 0.7776674 ,  0.02142365,  0.7776674 ,  0.02142365,  1.62339316,
                   -2.44769525,  1.62339316])
            >>> round(gd.bias, 6)
            0.799091
            >>> len(gd.losses)
            100
            >>> round(gd.losses[0], 6)
            0.693147
        """

        # encode the samples using the self.encoder
        # >>> YOUR CODE HERE >>>
        X_encoded = self.encoder.fit(X).encode(X)
        # <<< END OF YOUR CODE <<<

        # Initialize the weights to zeros (one per encoded feature), the bias
        # to 0.0, and reset the loss history.
        # >>> YOUR CODE HERE >>>
        self.weights = np.zeros(X_encoded.shape[1])
        self.bias = 0.0
        self.losses = []
        # <<< END OF YOUR CODE <<<

        for _ in range(self.n_iters):
            # Forward pass: predicted probabilities for the current parameters.
            # >>> YOUR CODE HERE >>>
            y_prob = self._sigmoid(X_encoded @ self.weights + self.bias)
            # <<< END OF YOUR CODE <<<

            # Record the loss before this update.
            # >>> YOUR CODE HERE >>>
            self.losses.append(self._compute_loss(y, y_prob))
            # <<< END OF YOUR CODE <<<

            # Compute the gradients and take a step in the negative gradient direction.
            # >>> YOUR CODE HERE >>>
            dw, db = self._compute_gradients(X_encoded, y, y_prob)
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            # <<< END OF YOUR CODE <<<

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability that each example has label 1.

        Parameters:
            X (np.ndarray): Examples to predict.

        Returns:
            np.ndarray: Predicted probabilities.

        Example:
            >>> gd = GradientDescent(learning_rate=0.5, n_iters=100)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> gd.fit(X, y)
            >>> gd.predict_proba(X)
            array([0.47669954, 0.92167541, 0.92167541, 0.47669954, 0.16717855])
        """

        assert self.weights is not None, "GradientDescent classifier has not been fit yet."

        # encode the samples using the self.encoder
        # >>> YOUR CODE HERE >>>
        X_encoded = self.encoder.encode(X)
        # <<< END OF YOUR CODE <<<

        # >>> YOUR CODE HERE >>>
        y_prob = self._sigmoid(X_encoded @ self.weights + self.bias)
        # <<< END OF YOUR CODE <<<
        return y_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given examples: 1 if the predicted
        probability is at least 0.5, otherwise 0.

        Parameters:
            X (np.ndarray): Examples to predict.

        Returns:
            np.ndarray: Predicted labels (ints).

        Example:
            >>> gd = GradientDescent(learning_rate=0.5, n_iters=100)
            >>> X = np.array([ \
                    ["A", "1", "XX"], \
                    ["B", "2", "YY"], \
                    ["B", "2", "NA"], \
                    ["A", "1", "XX"], \
                    ["B", "2", "XX"], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> gd.fit(X, y)
            >>> gd.predict(X)
            array([0, 1, 1, 0, 0])
        """

        # >>> YOUR CODE HERE >>>
        y_pred = (self.predict_proba(X) >= 0.5).astype(int)
        # <<< END OF YOUR CODE <<<
        return y_pred


if __name__ == "__main__":
    import doctest
    import os

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from utils import load_hw3_csv, print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X, y = load_hw3_csv(os.path.join(
            os.path.dirname(__file__), "loan_train.csv"))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        gd = GradientDescent(learning_rate=0.5, n_iters=500)
        gd.fit(X_train, y_train)
        print(
            f"Gradient Descent | Loss: {gd.losses[0]:.4f} -> {gd.losses[-1]:.4f}")
        y_pred = gd.predict(X_train)
        print(
            f"Gradient Descent | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")

        y_pred = gd.predict(X_test)
        print(
            f"Gradient Descent |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

        # sklearn's (unregularized) logistic regression as a reference
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit(X_train).encode(X_train)
        X_test_encoded = encoder.encode(X_test)
        lr = LogisticRegression(penalty=None, max_iter=1000)
        lr.fit(X_train_encoded, y_train)
        y_pred = lr.predict(X_train_encoded)
        print(
            f"sklearn LogisticRegression | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")
        y_pred = lr.predict(X_test_encoded)
        print(
            f"sklearn LogisticRegression |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    else:
        print_red("\nDoctests failed!\n")
