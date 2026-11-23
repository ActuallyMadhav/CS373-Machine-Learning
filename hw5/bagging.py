import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier


class Bagging(BaseEnsemble):
    """
    Bagging classifier with DecisionTreeClassifier as base learner.

    Attributes:
        learners: List of DecisionTreeClassifier objects.
        random_seed: Random seed for reproducibility.
    """

    def __init__(self, n_learners: int, random_seed: int = 42) -> None:
        """
        Constructor for Bagging classifier. Creates a list of DecisionTreeClassifier
        objects and sets the random seed.

        Parameters:
            n_learners (int): Number of learners to train.
            random_seed (int): Random seed for reproducibility.

        Returns:
            None

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> bagging
            Bagging(n_learners=10)
            >>> bagging.learners[0]
            DecisionTreeClassifier(random_state=42)
            >>> bagging.random_seed
            42
        """

        # >>> YOUR CODE HERE >>>
        self.random_seed = random_seed
        self.learners = [DecisionTreeClassifier(random_state=self.random_seed) for i in range(n_learners)]
        np.random.seed(self.random_seed)
        # <<< END OF YOUR CODE <<<

    def _prepare_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare a sample of data for a learner. This method is used to create a
        bootstrap sample with replacement.

        Parameters:
            X (np.ndarray): Features.

        Returns:
            np.ndarray: Indices of the sample.

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> bagging._prepare_sample(X)
            array([2, 0, 2])
        """

        # >>> YOUR CODE HERE >>>
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
        # <<< END OF YOUR CODE <<<
        return indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Bagging classifier.

        Parameters:
            X (np.ndarray): Training examples.
            y (np.ndarray): Training labels.

        Returns:
            None

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([ \
                    [ 0.5,  1.2], \
                    [-1.3,  0.4], \
                    [ 2.1, -0.7], \
                    [ 0.2,  0.9], \
                    [-0.8, -1.5], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> bagging.fit(X, y)
            >>> bagging.learners[0].predict(X)
            array([0, 0, 1, 0, 0])
        """

        # Train each learner on a bootstrap sample
        # >>> YOUR CODE HERE >>>
        for learner in self.learners:
            sample = self._prepare_sample(X)
            learner.fit(X[sample], y[sample])
        # <<< END OF YOUR CODE <<<

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given examples.

        Parameters:
            X (np.ndarray): Examples to predict.

        Returns:
            np.ndarray: Predicted labels.

        Example:
            >>> bagging = Bagging(n_learners=10)
            >>> X = np.array([ \
                    [ 0.5,  1.2], \
                    [-1.3,  0.4], \
                    [ 2.1, -0.7], \
                    [ 0.2,  0.9], \
                    [-0.8, -1.5], \
                ])
            >>> y = np.array([1, 1, 1, 0, 0])
            >>> bagging.fit(X, y)
            >>> bagging.predict(X)
            array([1, 1, 1, 0, 0]...)
        """

        assert hasattr(self.learners[0], "tree_"), "Bagging classifier has not been fit yet."

        # Predict the labels for each learner
        # >>> YOUR CODE HERE >>>
        predictions = np.array([learner.predict(X) for learner in self.learners]).astype(int)
        # <<< END OF YOUR CODE <<<

        # Count the number of votes for each label
        # >>> YOUR CODE HERE >>>
        counts = np.apply_along_axis(np.bincount, 0, predictions, minlength=predictions.max()+1)
        # <<< END OF YOUR CODE <<<

        # Return the label with the most votes
        # >>> YOUR CODE HERE >>>
        y_pred = np.argmax(counts, axis=0)
        # <<< END OF YOUR CODE <<<
        return y_pred

    def __repr__(self) -> str:
        """
        String representation of Bagging classifier.

        Parameters:
            None

        Returns:
            str: String representation of Bagging classifier.
        """
        return f"Bagging(n_learners={len(self.learners)})"


if __name__ == "__main__":
    import doctest
    import os

    from pca import PCA
    from sklearn.metrics import accuracy_score
    from utils import load_imagenette, print_green, print_red

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')

    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X_train, y_train, X_test, y_test = load_imagenette()

        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], 1024)
        X_test = X_test.reshape(X_test.shape[0], 1024)

        # Use the PCA components as features
        pca = PCA(n_components=50)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        bagging = Bagging(n_learners=10)
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_train)
        print(
            f"Bagging | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")

        y_pred = bagging.predict(X_test)
        print(
            f"Bagging |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

        # without bagging
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_train)
        print(
            f"Decision Tree | Train accuracy: {accuracy_score(y_train, y_pred):.4f}")
        y_pred = dt.predict(X_test)
        print(
            f"Decision Tree |  Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    else:
        print_red("\nDoctests failed!\n")
