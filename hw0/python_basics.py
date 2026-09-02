import random
from typing import Dict, List, Union


class Exponentiation:
    """This class takes a base and an integer power and computes base**power.

    Attributes:
        base (Union[int, float]): The base value.
        power (int): The exponent (must be an integer).
        result (Union[int, float]): The computed value (created after calling compute()).
    """

    def __init__(self, base: Union[int, float], power: int) -> None:
        """Constructor for Exponentiation.

        Saves the inputs as attributes. The exponent must be an integer.

        Parameters:
            base (Union[int, float]): The base value.
            power (int): The exponent (must be an integer).

        Raises:
            TypeError: If power is not an integer.

        Examples:
            >>> exp = Exponentiation(2, 3)
            >>> exp.base
            2
            >>> exp.power
            3
            >>> exp.result
            Traceback (most recent call last):
            ...
            AttributeError: 'Exponentiation' object has no attribute 'result'
        """

        if not isinstance(power, int):
            raise TypeError("power must be an integer")

        # >>> YOUR CODE HERE >>>
        self.base = base
        self.power = power
        # <<< END OF YOUR CODE <<<

    def compute(self) -> Union[int, float]:
        """Computes base**power and stores it in self.result.

        Returns:
            Union[int, float]: The computed value.

        Raises:
            ValueError: If base == 0 and power <= 0.

        Examples:
            >>> exp = Exponentiation(2, 3)
            >>> exp.compute()
            8
            >>> exp.result
            8
        """
        if self.base == 0 and self.power <= 0:
            raise ValueError("0 cannot be raised to a non-positive power")
        # >>> YOUR CODE HERE >>>
        self.result = self.base**self.power
        # <<< END OF YOUR CODE <<<

        return self.result

    def display(self) -> None:
        """Prints the stored values in the required format.

        The output format MUST match exactly:
            Base: <base with 2 decimals>
            Power: <power as integer>
            Result: <result with 4 decimals>

        Examples:
            >>> exp = Exponentiation(2, 3)
            >>> _ = exp.compute()
            >>> exp.display()
            Base: 2.00
            Power: 3
            Result: 8.0000
        """

        # >>> YOUR CODE HERE >>>
        print(f"Base: {self.base:.2f}")
        print(f"Power: {self.power}")
        print(f"Result: {self.result:.4f}")
        # <<< END OF YOUR CODE <<<


class DataStructure:
    def generate_list(self) -> List[int]:
        """Returns a list of 8 random integers between -5 and 5 (inclusive).

        IMPORTANT: Do NOT set random.seed() inside your solution.

        Returns:
            List[int]: A list of 8 random integers in [-5, 5].

        Examples:
            >>> data_structure = DataStructure()
            >>> random.seed(0)
            >>> data_structure.generate_list()
            [1, 1, -5, -1, 3, 2, 1, -1]
        """

        # >>> YOUR CODE HERE >>>
        my_list = [random.randint(-5, 5) for x in range(8)]
        # <<< END OF YOUR CODE <<<
        return my_list

    def find_non_negative(self, my_list: List[int]) -> set[int]:
        """Returns a set containing all values in my_list that are non-negative.

        Parameters:
            my_list (List[int]): A list of integers.

        Returns:
            set[int]: A set containing all values >= 0.

        Examples:
            >>> data_structure = DataStructure()
            >>> random_list = [1, 1, -5, -1, 3, 2, 1, -1]
            >>> data_structure.find_non_negative(random_list)
            {1, 2, 3}
        """

        # >>> YOUR CODE HERE >>>
        nonneg = {x for x in my_list if x >= 0}
        # <<< END OF YOUR CODE <<<
        return nonneg

    def count_signs(self, my_list: List[int]) -> Dict[str, int]:
        """Returns a dictionary counting negatives, zeros, and positives.

        The returned dictionary MUST have exactly these keys:
            - "neg": number of elements < 0
            - "zero": number of elements == 0
            - "pos": number of elements > 0

        Parameters:
            my_list (List[int]): A list of integers.

        Returns:
            Dict[str, int]: Dictionary with keys {"neg", "zero", "pos"}.

        Examples:
            >>> data_structure = DataStructure()
            >>> data_structure.count_signs([1, 1, -5, -1, 3, 2, 1, -1])
            {'neg': 3, 'zero': 0, 'pos': 5}
            >>> data_structure.count_signs([0, 0, 1, -1])
            {'neg': 1, 'zero': 2, 'pos': 1}
        """

        # >>> YOUR CODE HERE >>>
        counts = {}
        counts['neg'] = 0
        counts['zero'] = 0
        counts['pos'] = 0

        for i in my_list:
            if i > 0:
                counts['pos'] += 1
            elif i < 0:
                counts['neg'] += 1
            else:
                counts['zero'] += 1
        # <<< END OF YOUR CODE <<<
        return counts


if __name__ == "__main__":
    import doctest

    from utils import print_green, print_red

    # Run the doctests. If all tests pass, print "All tests pass!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
