import pandas as pd
import numpy as np
from typing import Tuple

def load_data(url):
    """
    Load the Pima Indians Diabetes Dataset.
    
    Args:
    url (str): Path or URL of the dataset CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=columns)
    return df

def standardize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize the training and test datasets.

    Standardization formula: z = (x - μ) / σ
    where z is the standardized value, x is the original value,
    μ is the mean, and σ is the standard deviation of the TRAINING set.
    If a feature has σ = 0, replace σ with 1e-7 to avoid division by zero.

    Parameters:
    X_train (numpy.ndarray): Training dataset
    X_test (numpy.ndarray): Test dataset

    Returns:
    tuple: A tuple containing the standardized training and test datasets

    Examples:
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X_train = np.random.rand(5, 3)
    >>> X_test = np.random.rand(2, 3)
    >>> X_train_std, X_test_std = standardize(X_train, X_test)
    >>> np.allclose(X_train_std.mean(axis=0), 0, atol=1e-6)
    True
    >>> np.allclose(X_train_std.std(axis=0), 1, atol=1e-6)
    True
    >>> X_train_std.shape == X_train.shape
    True
    >>> X_test_std.shape == X_test.shape
    True
    """
    # >>> YOUR CODE HERE >>>
    X_train_std = ...
    X_test_std = ...
    # <<< END OF YOUR CODE <<<
    
    return X_train_std, X_test_std

def preprocess_data(df):
    """
    Preprocess the dataset by splitting features and target.
    
    Args:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    tuple: X (features) and y (target) as numpy arrays
    """
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values
    return X, y

def split_data(X, y, test_size=0.25, random_state=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets. Should be reproducible with same random_state.

    You must implement the split with numpy only (no scikit-learn). Use exactly
    the following procedure so that the autograder can check your split:
      1. rng = np.random.default_rng(random_state)
      2. perm = rng.permutation(n_samples)          # a random ordering of the row indices
      3. n_test = int(np.ceil(test_size * n_samples))
      4. the first n_test entries of perm are the test indices,
         the remaining entries are the training indices (keep them in perm order)
    
    Args:
    X (numpy.ndarray): Features
    y (numpy.ndarray): Target
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility
    
    Returns:
    tuple: X_train, X_test, y_train, y_test
    
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=1)
    >>> X_train2, X_test2, y_train2, y_test2 = split_data(X, y, test_size=0.25, random_state=1)
    >>> X_train.shape
    (3, 2)
    >>> X_test.shape
    (1, 2)
    >>> y_train.shape
    (3,)
    >>> y_test.shape
    (1,)
    >>> np.allclose(X_train, X_train2) and np.allclose(X_test, X_test2) and np.allclose(y_train, y_train2) and np.allclose(y_test, y_test2)
    True
    >>> X_test
    array([[1, 2]])
    >>> y_train
    array([1, 0, 1])
    """
    # >>> YOUR CODE HERE >>>
    X_train = ...
    y_train = ...
    X_test = ...
    y_test = ...
    # <<< END OF YOUR CODE <<<
    return X_train, X_test, y_train, y_test

def accuracy(y_true, y_pred) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
    y_true (numpy.ndarray): True labels
    y_pred (numpy.ndarray): Predicted labels
    
    Returns:
    float: Accuracy score
    
    >>> y_true = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1])
    >>> accuracy(y_true, y_pred)
    0.75
    """
    # >>> YOUR CODE HERE >>>
    acc = ...
    # <<< END OF YOUR CODE <<<    
    return acc

if __name__ == "__main__":
    import doctest
    doctest.testmod()
