from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.figure import Figure
import random


class PyTorchCNN(nn.Module):
    """
    A PyTorch convolutional neural network for MNIST (handwritten digits).

    Attributes:
        num_classes (int): The number of classes.
        model (nn.Sequential): The model.
    """

    def __init__(self, num_classes: int = 10, random_seed: int = 42) -> None:
        """
        Initialize the PyTorchCNN model.

        Parameters:
            num_classes (int): The number of classes.
            random_seed (int): The random seed.

        Returns:
            None

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> pytorch_cnn.num_classes
            10
            >>> pytorch_cnn.model
            Sequential...
            >>> pytorch_cnn(torch.randn(2, 1, 28, 28)).shape
            torch.Size([2, 10])
        """
        super(PyTorchCNN, self).__init__()

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.num_classes = num_classes

        # Define the model that runs on MNIST (1 input channel, 28x28 images).
        # The model has 3 convolutional blocks:
        #   Block 1: Conv2d(1, 32, kernel_size=3, stride=1, padding=1) -> ReLU -> MaxPool2d(2) -> BatchNorm2d(32) -> Dropout(0.3)
        #   Block 2: Conv2d(32, 64, kernel_size=3, stride=1, padding=1) -> ReLU -> MaxPool2d(2) -> BatchNorm2d(64) -> Dropout(0.3)
        #   Block 3: Conv2d(64, 128, kernel_size=3, stride=1, padding=1) -> ReLU -> MaxPool2d(2) -> BatchNorm2d(128) -> Dropout(0.3)
        # Followed by a classifier:
        #   Flatten -> Linear(?, 256) -> ReLU -> Linear(256, num_classes)
        # You must work out the number of input features (?) of the first Linear
        # layer yourself: track the spatial size of the feature map through every
        # Conv2d and MaxPool2d layer (see the output-size formula in the handout).
        # >>> YOUR CODE HERE >>>
        self.model = nn.Sequential(
            # block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout(0.3),

            #block2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),

            #block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),

            #classifier
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
        # <<< END OF YOUR CODE <<<

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Parameters:
            x (torch.Tensor): The input data, of shape (N, 1, 28, 28).

        Returns:
            torch.Tensor: The output of the neural network (logits), of shape (N, num_classes).
        """
        output = self.model(x)
        return output

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters (weights and biases) of the model.
        Your code must work for any model, not only for the architecture above.

        Returns:
            int: The total number of trainable parameters, as a Python int.

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> isinstance(pytorch_cnn.count_parameters(), int)
            True
        """
        # Hint: self.parameters() yields every parameter tensor of the model;
        # p.numel() is the number of entries of tensor p, and p.requires_grad
        # tells you whether p is trainable.
        # >>> YOUR CODE HERE >>>
        num_parameters = 0
        for p in self.parameters():
            if p.requires_grad:
                num_parameters += p.numel()
    
        # <<< END OF YOUR CODE <<<
        return int(num_parameters)

    def _evaluate(self,
                  X: torch.Tensor,
                  y: torch.Tensor,
                  loss_function: nn.Module,
                  batch_size: int = 1000) -> Tuple[float, float]:
        """
        Compute the average loss and the accuracy on (X, y) in batches to
        limit intermediate activation memory. X and y are already in memory.

        Returns:
            Tuple[float, float]: (average loss, accuracy)
        """
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in zip(X.split(batch_size), y.split(batch_size)):
                output = self.forward(X_batch)
                total_loss += loss_function(output, y_batch).item() * X_batch.shape[0]
                correct += (output.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()
        return total_loss / X.shape[0], correct / X.shape[0]

    def fit(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            X_val: torch.Tensor,
            y_val: torch.Tensor,
            epochs: int = 10,
            learning_rate: float = 0.001) -> None:
        """
        Train the neural network.

        (This method is called fit and not train because nn.Module already has
        a method train(mode), which PyTorch itself calls inside .eval().)

        Parameters:
            X_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels (one-hot encoded).
            X_val (torch.Tensor): The validation data.
            y_val (torch.Tensor): The validation labels (one-hot encoded).
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate.

        Returns:
            None
        """

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        batch_size = 64

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            indices = torch.randperm(X_train.shape[0])
            for start in range(0, X_train.shape[0], batch_size):
                idx = indices[start:start+batch_size]
                X_batch = X_train[idx]
                y_batch = y_train[idx]

                optimizer.zero_grad()
                output = self.forward(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()

            train_loss, train_acc = self._evaluate(X_train, y_train, loss_function)
            val_loss, val_acc = self._evaluate(X_val, y_val, loss_function)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"[{epoch+1:} / {epochs}] | Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.5f} | Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}")

        self.train_val_metrics = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs
        }

    def predict(self, X: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        """
        Predict the class labels for the input data.

        Parameters:
            X (torch.Tensor): The input data.
            batch_size (int): The number of images processed at once.

        Returns:
            torch.Tensor: The predicted class labels.

        Examples:
            >>> pytorch_cnn = PyTorchCNN()
            >>> X = torch.randn(10, 1, 28, 28)
            >>> pytorch_cnn.predict(X).shape
            torch.Size([10])
        """

        # Set the model to eval mode and use torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            return torch.cat([self.forward(X_batch).argmax(dim=1) for X_batch in X.split(batch_size)])

    def plot_train_val_metrics(self) -> Tuple[Figure, np.ndarray]:
        """
        Plot the training and validation metrics.

        Parameters:
            None

        Returns:
            Tuple[Figure, np.ndarray]: A tuple containing the matplotlib
                Figure and Axes objects.
        """

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the training and validation losses
        ax[0].plot(self.train_val_metrics["train_losses"], label="Train Loss")
        ax[0].plot(self.train_val_metrics["val_losses"], label="Val Loss")

        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        # Plot the training and validation accuracies
        ax[1].plot(self.train_val_metrics["train_accs"], label="Train Accuracy")
        ax[1].plot(self.train_val_metrics["val_accs"], label="Val Accuracy")

        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()

        fig.suptitle("Train/Val Metrics")
        fig.tight_layout()

        plt.savefig("pytorch_cnn_train_val_metrics.png")

        return fig, ax

    def report_my_judgement(self) -> str:
        """
        Return your judgement on whether the neural network is overfitting or not.
        **You should not write any other code in this method.**

        Returns:
            str: Your judgement ("overfitting", "underfitting", or "generalizing")

        """

        # >>> YOUR CODE HERE >>>
        my_judgement = "generalizing"
        # <<< END OF YOUR CODE <<<

        assert my_judgement in ["overfitting", "underfitting", "generalizing"], \
            f"Your judgement must be one of 'overfitting', 'underfitting', or 'generalizing'. You entered: {my_judgement}\n"

        return my_judgement

    def accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the accuracy of the neural network on the input data.

        Parameters:
            X (torch.Tensor): The input data.
            y (torch.Tensor): The true class labels (one-hot encoded).

        Returns:
            float: The accuracy of the neural network.
        """
        assert X.shape[0] == y.shape[0], f"X.shape[0] != y.shape[0] ({X.shape[0]} != {y.shape[0]})"
        correct = torch.sum(self.predict(X) == y.argmax(dim=1))
        return (correct / X.shape[0]).item()


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

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            print_blue("GPU not available. Using CPU instead.")
        print_blue(f"Using device: {device}\n")

        X_train, y_train, X_test, y_test = load_mnist(return_tensor=True)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=10000, test_size=1000, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test, y_test, train_size=1000, test_size=1, random_state=42)

        # Add the channel dimension: (N, 28, 28) -> (N, 1, 28, 28)
        X_train = X_train.unsqueeze(1).float().to(device)
        X_val = X_val.unsqueeze(1).float().to(device)
        X_test = X_test.unsqueeze(1).float().to(device)

        # Normalize the images
        X_train = X_train / 255
        X_val = X_val / 255
        X_test = X_test / 255

        # One-hot encode the labels
        y_train = F.one_hot(y_train, num_classes=10).float().to(device)
        y_val = F.one_hot(y_val, num_classes=10).float().to(device)
        y_test = F.one_hot(y_test, num_classes=10).float().to(device)

        # Train the neural network
        pytorch_cnn = PyTorchCNN(num_classes=10).to(device)
        print_blue(f"Number of trainable parameters: {pytorch_cnn.count_parameters()}\n")
        pytorch_cnn.fit(X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.001)

        test_accuracy = pytorch_cnn.accuracy(X_test, y_test)

        pytorch_cnn.plot_train_val_metrics()

        assert_greater_equal(test_accuracy, 0.94, f"\nAccuracy on test set: {test_accuracy:.5f}\n")

        print_blue(f"My judgement: {pytorch_cnn.report_my_judgement()}\n")
    else:
        print_red("\nDoctests failed!\n")
