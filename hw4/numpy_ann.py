import numpy as np


class NumpyANN():
    """
    A NumPy neural network with two hidden layers and tanh-approximate GELU.

    Attributes:
        input_size (int): The number of input features.
        hidden_size1 (int): The number of hidden units in the first hidden layer.
        hidden_size2 (int): The number of hidden units in the second hidden layer.
        output_size (int): The number of output units.
        lr (float): The learning rate.
        W1 (np.ndarray): The weights of the first layer.
        b1 (np.ndarray): The biases of the first layer.
        W2 (np.ndarray): The weights of the second layer.
        b2 (np.ndarray): The biases of the second layer.
        W3 (np.ndarray): The weights of the third layer.
        b3 (np.ndarray): The biases of the third layer.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size1: int,
                 hidden_size2: int,
                 output_size: int,
                 lr: float,
                 random_seed: int = 42) -> None:
        """
        Constructor for the NumpyANN class.

        Parameters:
            input_size (int): The number of input features.
            hidden_size1 (int): The number of hidden units in the first hidden layer.
            hidden_size2 (int): The number of hidden units in the second hidden layer.
            output_size (int): The number of output units.
            lr (float): The learning rate.
            random_seed (int): The random seed to use. Defaults to 42.

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> ann.input_size
            5
            >>> ann.hidden_size1
            3
            >>> ann.hidden_size2
            4
            >>> ann.output_size
            2
            >>> ann.lr
            0.01
            >>> ann.W3.shape
            (4, 2)
            >>> ann.b3.shape
            (1, 2)
        """

        # Set the random seed
        np.random.seed(random_seed)

        # Initialize attributes
        # >>> YOUR CODE HERE >>>
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.lr = lr
        # <<< END OF YOUR CODE <<<

        # Initialize weight for the first layer (input -> hidden1)
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.01
        self.b1 = np.zeros((1, self.hidden_size1))

        # Initialize weight for the second layer (hidden1 -> hidden2) like the first layer
        # >>> YOUR CODE HERE >>>
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.01
        self.b2 = np.zeros((1, self.hidden_size2))
        # <<< END OF YOUR CODE <<<

        # Initialize weight for the third layer (hidden2 -> output) like the first layer
        # >>> YOUR CODE HERE >>>
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))
        # <<< END OF YOUR CODE <<<

    def _gelu(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the tanh approximation to GELU from Section 1.2.
        Apply the formula element-wise without changing the input shape.

        Parameters:
            z (np.ndarray): The input to the GELU function.

        Returns:
            np.ndarray: The GELU activations.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> np.round(ann._gelu(np.array([-1., 0., 1.])), 6)
            array([-0.158808,  0.      ,  0.841192])
        """
        # >>> YOUR CODE HERE >>>
        u = np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
        gelu = 0.5 * z * (1 + np.tanh(u))
        # <<< END OF YOUR CODE <<<
        return gelu

    def _gelu_grad(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the tanh GELU approximation in Section 1.3.

        Parameters:
            z (np.ndarray): The preactivations, not the GELU outputs.

        Returns:
            np.ndarray: The element-wise GELU derivatives.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> np.round(ann._gelu_grad(np.array([-1., 0., 1.])), 6)
            array([-0.082964,  0.5     ,  1.082964])
        """
        # >>> YOUR CODE HERE >>>
        u = np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)
        gelu_grad = 0.5*(1 + np.tanh(u)) + 0.5*z*(1 - (np.tanh(u)**2))*(np.sqrt(2/np.pi)*(1 + 0.134145*z**2))
        # <<< END OF YOUR CODE <<<
        return gelu_grad

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the softmax activation function.

        Parameters:
            z (np.ndarray): The input to the softmax function.

        Returns:
            np.ndarray: The output of the softmax function.
        """

        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes mean cross-entropy over the batch. Clip predicted probabilities
        to [1e-12, 1.0] before taking logarithms for numerical stability.

        Parameters:
            y_pred (np.ndarray): The predicted probabilities for each class.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            float: The cross entropy loss.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> float(round(ann._cross_entropy(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[1, 0], [0, 1]])), 5))
            0.69315
        """
        # >>> YOUR CODE HERE >>>
        y_pred = np.clip(y_pred, 1e-12, 1.0)
        cross_entropy = (-1 / y_true.shape[0]) * np.sum(y_true * np.log(y_pred))
        # <<< END OF YOUR CODE <<<
        return cross_entropy

    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Performs a forward pass through the network.

        Parameters:
            X (np.ndarray): The input to the network.
            training (bool): Whether the network is in training mode or not. Defaults to True.

        Returns:
            np.ndarray: The output of the network.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> ann._forward(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50000143, 0.49999857]])
        """

        # Forward pass, a = activation(z), z = XW + b
        # Layer 1, use GELU activation
        # >>> YOUR CODE HERE >>>
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._gelu(z1)
        # <<< END OF YOUR CODE <<<

        # Layer 2, use GELU activation
        # >>> YOUR CODE HERE >>>
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._gelu(z2)
        # <<< END OF YOUR CODE <<<

        # Layer 3, use softmax activation
        # >>> YOUR CODE HERE >>>
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._softmax(z3)
        # <<< END OF YOUR CODE <<<

        # Save the activations for backpropagation
        if training:
            self.z1 = z1
            self.a1 = a1
            self.z2 = z2
            self.a2 = a2
            self.z3 = z3
            self.a3 = a3

        return a3

    def _backward(self, X: np.ndarray, y_true: np.ndarray) -> None:
        """
        Performs a backward propagation through the network.

        Parameters:
            X (np.ndarray): The input to the network.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> ann._forward(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50000143, 0.49999857]])
            >>> ann._backward(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]))
        """

        # Get the number of samples
        m = y_true.shape[0]

        # Backward pass
        # Unaveraged per-example errors; d(mean loss)/dz3 = dz3 / m.
        dz3 = self.a3 - y_true

        # dL/dW3 = a2.T @ dz3 / m
        dW3 = np.dot(self.a2.T, dz3) / m

        # dL/db3
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        # Unaveraged error: dz2 = (dz3 @ W3.T) * gelu_grad(z2)
        # >>> YOUR CODE HERE >>>
        dz2 = np.dot(dz3, self.W3.T) * self._gelu_grad(self.z2)
        # <<< END OF YOUR CODE <<<

        # dL/dW2 = a1.T @ dz2 / m
        # >>> YOUR CODE HERE >>>
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        # <<< END OF YOUR CODE <<<

        # dL/db2
        # >>> YOUR CODE HERE >>>
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        # <<< END OF YOUR CODE <<<

        # Unaveraged error: dz1 = (dz2 @ W2.T) * gelu_grad(z1)
        # >>> YOUR CODE HERE >>>
        dz1 = np.dot(dz2, self.W2.T) * self._gelu_grad(self.z1)
        # <<< END OF YOUR CODE <<<

        # dL/dW1 = X.T @ dz1 / m
        # dL/db1
        # >>> YOUR CODE HERE >>>
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        # <<< END OF YOUR CODE <<<

        # Perform gradient descent for layer 3
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        # Perform gradient descent for layer 2
        # >>> YOUR CODE HERE >>>
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        # <<< END OF YOUR CODE <<<

        # Perform gradient descent for layer 1
        # >>> YOUR CODE HERE >>>
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        # <<< END OF YOUR CODE <<<

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int,
              batch_size: int) -> None:
        """
        Trains the neural network.

        Parameters:
            X_train (np.ndarray): The training data.
            y_train (np.ndarray): The training labels.
            X_val (np.ndarray): The validation data.
            y_val (np.ndarray): The validation labels.
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size to use for mini-batch gradient descent.

        Returns:
            None

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> ann.train(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]), np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]]), epochs=1, batch_size=1)
        """
        for epoch in range(epochs):
            # Prepare the batches for mini-batch gradient descent
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # Perform a forward pass
                # >>> YOUR CODE HERE >>>
                self._forward(X_batch, training=True)
                # <<< END OF YOUR CODE <<<

                # Perform a backward pass
                # >>> YOUR CODE HERE >>>
                self._backward(X_batch, y_batch)
                # <<< END OF YOUR CODE <<<

            y_pred_train = self._forward(X_train, training=False)
            y_pred_val = self._forward(X_val, training=False)

            # Compute the loss
            # >>> YOUR CODE HERE >>>
            train_loss = self._cross_entropy(y_pred_train, y_train)
            val_loss = self._cross_entropy(y_pred_val, y_val)
            # <<< END OF YOUR CODE <<<

            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)

            if (epoch+1) % 10 == 0:
                print(f"[{epoch+1:} / {epochs}] | Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the neural network.

        Parameters:
            X (np.ndarray): The input to the network.

        Returns:
            np.ndarray: The predicted probabilities for each class.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> ann.predict(np.array([[1, 2, 3, 4, 5]]))
            array([[0.50000143, 0.49999857]])
        """

        # >>> YOUR CODE HERE >>>
        pred = self._forward(X, training=False)
        # <<< END OF YOUR CODE <<<
        return pred

    def accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the accuracy of the network.

        Parameters:
            X (np.ndarray): The input to the network.
            y_true (np.ndarray): The ground truth labels (one-hot encoded)

        Returns:
            float: The accuracy of the network.

        Examples:
            >>> ann = NumpyANN(input_size=5, hidden_size1=3, hidden_size2=4, output_size=2, lr=0.01)
            >>> float(ann.accuracy(np.array([[1, 2, 3, 4, 5]]), np.array([[1, 0]])))
            1.0
        """

        # >>> YOUR CODE HERE >>>
        y_pred = self.predict(X)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
        # <<< END OF YOUR CODE <<<
        return acc

if __name__ == "__main__":
    import doctest
    import os

    from sklearn.model_selection import train_test_split
    from utils import (assert_greater_equal, load_mnist, print_blue,
                       print_green, print_red)

    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green(f"\nDoctests passed!\n")

        X_train, y_train, X_test, y_test = load_mnist(return_tensor=False)

        # Sample 2000 training, 200 validation, and 200 test images.
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=2000, test_size=200, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=200, test_size=1, random_state=42)
        # Flatten the images
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_val = X_val.reshape(X_val.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        y_test = np.eye(10)[y_test]

        # Train the GELU network with learning rate 0.1.
        print_blue("Learning rate = 0.1")
        ann = NumpyANN(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10, lr=0.1)
        ann.train(X_train, y_train, X_val, y_val, epochs=250, batch_size=64)
        test_accuracy = ann.accuracy(X_test, y_test)
        assert_greater_equal(test_accuracy, 0.88, f"\nAccuracy on test set: {test_accuracy:.5f}\n")

        # The same constructor seed gives the same initial weights for both runs.
        print_blue("Learning rate = 0.01")
        ann_small_lr = NumpyANN(input_size=784, hidden_size1=128, hidden_size2=64, output_size=10, lr=0.01)
        ann_small_lr.train(X_train, y_train, X_val, y_val, epochs=250, batch_size=64)
        test_accuracy_small_lr = ann_small_lr.accuracy(X_test, y_test)
        print_blue(f"\nlr = 0.1  | test accuracy: {test_accuracy:.5f}")
        print_blue(f"lr = 0.01 | test accuracy: {test_accuracy_small_lr:.5f}\n")
    else:
        print_red("\nDoctests failed!\n")
