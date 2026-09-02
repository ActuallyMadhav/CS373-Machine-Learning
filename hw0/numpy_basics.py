import numpy as np
from typing import Tuple, Union


class NumpyBasics:
    """A collection of small NumPy exercises.

    Unless otherwise specified, DO NOT use any loops (for/while) in your solutions.
    """

    def split_even_odd_values(self, x: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Splits the input into even-valued and odd-valued elements.

        This function takes a 1-D list or NumPy array and returns a tuple of two NumPy arrays:
            - first: all even-valued elements in x
            - second: all odd-valued elements in x

        You MUST preserve the original order of elements in each output array.

        Parameters:
            x (Union[list, np.ndarray]): A 1-D list or NumPy array of integers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (even_values, odd_values)

        Examples:
            >>> nb = NumpyBasics()
            >>> a, b = nb.split_even_odd_values([1, 2, 3, 4, 5, 6])
            >>> a
            array([2, 4, 6])
            >>> b
            array([1, 3, 5])
        """
        x = np.array(x)

        # >>> YOUR CODE HERE >>>
        even_values = x[x % 2 == 0]
        odd_values = x[x % 2 != 0]
        # <<< END OF YOUR CODE <<<
        return even_values, odd_values

    def regularized_transpose_product_inverse(self, x: np.ndarray, lam: float = 1e-3) -> np.ndarray:
        """
        This function takes a 2-D numpy array and returns the inverse of the
        regularized product of the array and its transpose:

            (x * x^T + lam * I)^{-1}

        If the inverse does not exist, it raises a `LinAlgError`.

        Parameters:
            x (np.ndarray): A 2-D numpy array.
            lam (float): A non-negative regularization value.

        Returns:
            np.ndarray: A 2-D numpy array containing the inverse of the regularized
                        product of the array and its transpose. Feel free to use np.linalg.inv
                        to compute the inverse of matrix.

        Raises:
            np.linalg.LinAlgError: If the matrix cannot be inverted.

        Examples:
            >>> nb = NumpyBasics()

            # With lam=0, this matches the unregularized inverse.
            >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            >>> a = nb.regularized_transpose_product_inverse(x, lam=0.0)
            >>> type(a)
            <class 'numpy.ndarray'>
            >>> a
            array([[ 1.42592593, -0.59259259],
                   [-0.59259259,  0.25925926]])

            >>> x = np.array([[1, 1], [2, 3]])
            >>> a = nb.regularized_transpose_product_inverse(x, lam=0.0)
            >>> a
            array([[13., -5.],
                   [-5.,  2.]])

            # With lam>0, compare to NumPy's reference computation.
            >>> x = np.array([[1., 2.], [3., 4.], [5., 6.]])
            >>> out = nb.regularized_transpose_product_inverse(x, lam=1.0)
            >>> out
            array([[ 0.78448276, -0.1637931 , -0.11206897],
                   [-0.1637931 ,  0.71551724, -0.40517241],
                   [-0.11206897, -0.40517241,  0.30172414]])
        """

        # >>> YOUR CODE HERE >>>
        matrix = x @ np.transpose(x)
        identityMatrix = np.identity(matrix.shape[0])
        result = np.linalg.inv(matrix + lam*identityMatrix)
        # <<< END OF YOUR CODE <<<

        return result

    def nonnegative_max_normalization(self, x: np.ndarray) -> np.ndarray:
        """Replaces negative values with 0 and normalizes by the maximum value.

        Steps:
            1) Replace all negative values with 0.
            2) Divide the updated array by its maximum value.
               If the maximum value is 0, return an array of zeros.

        Parameters:
            x (np.ndarray): A 1-D NumPy array.

        Returns:
            np.ndarray: A 1-D NumPy array.

        Notes:
            - DO NOT use any loops.

        Examples:
            >>> nb = NumpyBasics()
            >>> x = np.array([-2., 0., 2., 4.])
            >>> y = nb.nonnegative_max_normalization(x)
            >>> y
            array([0. , 0. , 0.5, 1. ])
            >>> nb.nonnegative_max_normalization(np.array([-1., -2.]))
            array([0., 0.])
            >>> nb.nonnegative_max_normalization(np.array([0., 0., 0.]))
            array([0., 0., 0.])
        """

        # >>> YOUR CODE HERE >>>
        replaced = np.where(x < 0, 0, x)
        normalized = np.zeros(len(x))
        if np.max(replaced) != 0:
            normalized = replaced / np.max(replaced)
            
        # <<< END OF YOUR CODE <<<
        return normalized


if __name__ == "__main__":
    import doctest

    from utils import print_green, print_red

    # Run the doctests. If all tests pass, print "All tests pass!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
