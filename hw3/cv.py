from typing import List, Tuple

import numpy as np
from gradient_descent import GradientDescent
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(x): return x


class CrossValidation():
    """
    Cross validation.

    Attributes:
        k (int): Number of folds.
        seed (int): Random seed for reproducibility.    
    """

    def __init__(self, k: int = 5, random_seed: int = 42) -> None:
        """
        Constructor for CrossValidation. Sets the number of folds and the random seed.

        Parameters:
            k (int): Number of folds.
            random_seed (int): Random seed for reproducibility.

        Returns:
            None

        Example:
            >>> cv = CrossValidation(k=5)
            >>> cv
            CrossValidation(k=5, seed=42)
        """

        assert k > 1, "k must be greater than 1."

        # >>> YOUR CODE HERE >>>
        self.k = k
        self.random_seed = random_seed
        self.params = None
        self.scores = []
        self.models = []
        # <<< END OF YOUR CODE <<<

    def split(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Split the data into k folds.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: X_folds, y_folds

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 0, 1])
            >>> X_folds, y_folds = cv.split(X, y)
            >>> X_folds
            [array([[3, 4]]), array([[7, 8]]), array([[1, 2]]), array([[5, 6]])...]
            >>> y_folds
            [array([1]), array([1]), array([0]), array([0])...]
        """

        assert len(X) == len(y), "X and y must have the same length."
        assert self.k <= len(
            X), "k cannot be larger than the number of samples."

        # Use np.random.seed(self.random_seed) to set the random seed.
        # >>> YOUR CODE HERE >>>
        np.random.seed(self.random_seed)
        # <<< END OF YOUR CODE <<<


        # Shuffle the data using np.random.permutation.
        # >>> YOUR CODE HERE >>>
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        # <<< END OF YOUR CODE <<<


        # Split the data into k folds using np.array_split.
        # >>> YOUR CODE HERE >>>
        X_folds = np.array_split(X, self.k)
        y_folds = np.array_split(y, self.k)
        # <<< END OF YOUR CODE <<<



        return X_folds, y_folds

    def _combine_fold(self,
                      X_folds: List[np.ndarray],
                      y_folds: List[np.ndarray],
                      i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Use the ith fold as the test (validation) set and the rest as the training set.

        Parameters:
            X_folds (List[np.ndarray]): Features split into k folds.
            y_folds (List[np.ndarray]): Labels split into k folds.
            i (int): Index of the test (validation) fold.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X_folds = [np.array([[1, 2]]), np.array([[3, 4]]), np.array([[5, 6]]), np.array([[7, 8]])]
            >>> y_folds = [np.array([0]), np.array([1]), np.array([0]), np.array([1])]
            >>> cv._combine_fold(X_folds, y_folds, 0)
            (array([[3, 4],
                   [5, 6],
                   [7, 8]]), array([1, 0, 1]), array([[1, 2]]), array([0]))
        """

        # Use np.concatenate to combine the folds into the training set and the test set.
        # >>> YOUR CODE HERE >>>
        X_train = np.concatenate(X_folds[:i] + X_folds[i+1:])
        y_train = np.concatenate(y_folds[:i] + y_folds[i+1:])
        # <<< END OF YOUR CODE <<<



        # Assign the ith fold to be the test set.
        # >>> YOUR CODE HERE >>>
        X_test = X_folds[i]
        y_test = y_folds[i]
        # <<< END OF YOUR CODE <<<


        return X_train, y_train, X_test, y_test

    def score(self, X: np.ndarray, y: np.ndarray, model: GradientDescent) -> float:
        """
        Get the accuracy score of the model on the given data.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            model (GradientDescent): Model to evaluate.

        Returns:
            float: Accuracy score.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            >>> y = np.array([0, 1, 0, 1])
            >>> model = GradientDescent()
            >>> model.fit(X, y)
            >>> cv.score(X, y, model)
            1.0
        """
        # >>> YOUR CODE HERE >>>
        y_pred = model.predict(X)
        accuracy = float(np.mean(y_pred == y))
        # <<< END OF YOUR CODE <<<

        return accuracy

    def cross_validate(self, X: np.ndarray, y: np.ndarray, model: GradientDescent) -> Tuple[float, List[float]]:
        """
        Perform k-fold cross validation on the given model and data.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            model (GradientDescent): Model to evaluate.

        Returns:
            float: Average accuracy score across all folds.
            List[float]: Accuracy scores for each fold.

        Example:
            (Run this file or submit to Gradescope to run the test)
        """

        # Split the data into k folds.
        # >>> YOUR CODE HERE >>>
        X_folds, y_folds = self.split(X, y)
        # <<< END OF YOUR CODE <<<
        scores = []

        # For each fold, train the model on the training set and evaluate on the test set.
        for i in range(self.k):
            # Get the training set and the test set, using the ith fold as the test set.
            # >>> YOUR CODE HERE >>>
            X_train, y_train, X_test, y_test = self._combine_fold(X_folds, y_folds, i)
            # <<< END OF YOUR CODE <<<

            # Train the model on the training set.
            # >>> YOUR CODE HERE >>>
            model.fit(X_train, y_train)
            # <<< END OF YOUR CODE <<<

            # Evaluate the model on the test set, and append the score to the scores list.
            # >>> YOUR CODE HERE >>>
            scores.append(self.score(X_test, y_test, model))
            # <<< END OF YOUR CODE <<<

        # Return the average score across all folds.
        return np.mean(scores), scores

    def get_best_model(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       params: List[float]) -> Tuple[GradientDescent, float, List[float]]:
        """
        Perform k-fold cross validation of a GradientDescent model for each
        learning rate in params, and return the best model.

        Parameters:
            X (np.ndarray): Features.
            y (np.ndarray): Labels.
            params (List[float]): List of learning rates to try.

        Returns:
            GradientDescent: Best model.
            float: Best learning rate.
            scores (List[float]): Accuracy scores for each learning rate.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([["B", "B"], ["A", "A"], ["B", "A"], ["B", "A"], \
                              ["A", "B"], ["A", "A"], ["A", "B"], ["B", "B"]])
            >>> y = np.array([1, 0, 0, 0, 1, 0, 1, 0])
            >>> params = [0.01, 0.1, 1.0, 10.0]
            >>> model, best_lr, scores = cv.get_best_model(X, y, params)
            >>> best_lr
            1.0
            >>> scores
            [0.375, 0.625, 0.875, 0.625]
        """

        assert len(params) > 0, "params must be a non-empty list"

        # Store the parameters.
        self.params = params

        self.scores = []
        self.models = []

        # For each learning rate, train the model on the training set
        # and evaluate on the test set.
        for lr in tqdm(params):
            # Create a new model with the given learning rate and keep the default n_iters.
            # >>> YOUR CODE HERE >>>
            model = GradientDescent(learning_rate=lr)
            # <<< END OF YOUR CODE <<<

            # Perform k-fold cross validation on the model.
            # >>> YOUR CODE HERE >>>
            score, scores = self.cross_validate(X, y, model)
            score = float(score)
            # <<< END OF YOUR CODE <<<

            # Append the score to the scores list.
            self.scores.append(score)

            # Append the model to the models list.
            self.models.append(model)

        # Get the best learning rate and the corresponding model.
        # >>> YOUR CODE HERE >>>
        bestIndex = np.argmax(self.scores)
        best_lr = self.params[bestIndex]
        best_model = self.models[bestIndex]
        # <<< END OF YOUR CODE <<<

        return best_model, best_lr, self.scores

    def plot_learning_curve(self) -> Figure:
        """
        Plot the learning curve for the best model.

        Parameters:
            None

        Returns:
            Matplotlib figure: Learning curve.

        Example:
            >>> cv = CrossValidation(k=4)
            >>> X = np.array([["B", "B"], ["A", "A"], ["B", "A"], ["B", "A"], \
                              ["A", "B"], ["A", "A"], ["A", "B"], ["B", "B"]])
            >>> y = np.array([1, 0, 0, 0, 1, 0, 1, 0])
            >>> params = [0.01, 0.1, 1.0, 10.0]
            >>> model, best_lr, scores = cv.get_best_model(X, y, params)
            >>> cv.plot_learning_curve() # should also save the figure to learning_curve.png
            <Figure size ... with 1 Axes>
        """

        assert self.models is not None, "get_best_model must be called before plot_learning_curve"
        assert self.scores is not None, "get_best_model must be called before plot_learning_curve"
        assert self.params is not None, "get_best_model must be called before plot_learning_curve"

        plt.figure()

        # Plot the learning curve. The learning rates span several orders of
        # magnitude, so use a log scale on the x-axis.
        # >>> YOUR CODE HERE >>>
        plt.plot(self.params, self.scores, marker="o")
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Mean CV accuracy")
        plt.title("Learning curve: CV accuracy vs. learning rate")
        plt.savefig("learning_curve.png")
        # <<< END OF YOUR CODE <<<
        return plt.gcf()

    def __repr__(self) -> str:
        return f"CrossValidation(k={self.k}, seed={self.random_seed})"


if __name__ == "__main__":
    import doctest
    import os

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

        cv = CrossValidation(k=5)

        # tune the learning rate of gradient descent
        best_gd, best_lr, scores = cv.get_best_model(
            X_train, y_train, [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0])

        print(f"Gradient Descent | Best learning rate: {best_lr}")
        print(f"Gradient Descent | Best CV accuracy: {np.max(scores):.4f}\n")

        y_pred = best_gd.predict(X_train)
        print(
            f"Gradient Descent | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")

        y_pred = best_gd.predict(X_test)
        print(
            f"Gradient Descent |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

        cv.plot_learning_curve()

    else:
        print_red("\nDoctests failed!\n")
