import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataBasics:
    def __init__(self, filename: str) -> None:
        """
        The constructor for DataBasics class, which loads the CSV file into a
        pandas DataFrame and saves it to `self.df`.

        Parameters:
            filename (str): The filename of the csv file.

        Returns:
            None

        Examples:
            >>> db = DataBasics(os.path.join(os.path.dirname(__file__), \
                "exam_original.csv"))
            >>> db.df.shape
            (30641, 9)
            >>> columns = ["MathScore", "ReadingScore", "WritingScore"]
            >>> [int(db.df.loc[0, column]) for column in columns]
            [72, 72, 74]
            >>> type(db.df)
            <class 'pandas.core.frame.DataFrame'>
        """

        # >>> YOUR CODE HERE >>>
        self.df = pd.read_csv(filename)
        # <<< END OF YOUR CODE <<<


    def preprocess(self) -> None:
        """
        This function preprocesses the dataframe. It drops the "Unnamed: 0"
        column and renames the "ParentEduc" column to "ParentEducation". It
        also prints the first 3 rows of the preprocessed dataframe.

        Parameters:
            None

        Returns:
            None

        Examples:
            >>> db = DataBasics(os.path.join(os.path.dirname(__file__), \
                "exam_original.csv"))
            >>> import contextlib
            >>> import io
            >>> output = io.StringIO()
            >>> with contextlib.redirect_stdout(output):
            ...     db.preprocess()
            >>> db.df.shape
            (30641, 8)
            >>> "Unnamed: 0" in db.df.columns
            False
            >>> "ParentEducation" in db.df.columns
            True
        """

        # >>> YOUR CODE HERE >>>
        self.df.drop(columns=['Unnamed: 0'], inplace=True)
        self.df.rename(columns={'ParentEduc':'ParentEducation'}, inplace=True)
        print(self.df.head(3))
        # <<< END OF YOUR CODE <<<


    def writing_stats_given_score(self, score: int) -> Tuple[float, float, float]:
        """
        This function calculates the minimum, maximum, and median of the Math
        scores among students who scored less than or equal to `score` in writing.
        The function returns these statistics as a tuple of floats.

        Parameters:
            score (int): The maximum score to consider.

        Returns:
            Tuple[float, float, float]: The minimum, maximum, and median of the math
                scores.

        Examples:
            >>> db = DataBasics(os.path.join(os.path.dirname(__file__), "exam_original.csv"))
            >>> minimum, maximum, median = db.writing_stats_given_score(60)
            >>> minimum
            0.0
            >>> maximum
            84.0
            >>> median
            53.0
        """

        # >>> YOUR CODE HERE >>>
        # Filter rows where WritingScore is less than or equal to the given score
        filtered_df = self.df[self.df['WritingScore'] <= score]

        # Calculate the minimum, maximum, and median of the MathScore
        minimum = filtered_df['MathScore'].min()
        maximum = filtered_df['MathScore'].max()
        median = filtered_df['MathScore'].median()
        # <<< END OF YOUR CODE <<<

        return float(minimum), float(maximum), float(median)


    def generate_reading_score_histogram(self) -> plt.Figure:
        """
        This function generates a histogram of the reading scores. The
        histogram should have 15 bins and include appropriate labels and title.
        The histogram is saved as a png file and returned as a matplotlib
        Figure object.

        Parameters:
            None

        Returns:
            plt.Figure: The histogram of the reading scores.

        Examples:
            >>> db = DataBasics(os.path.join(os.path.dirname(__file__), \
                "exam_original.csv"))
            >>> fig = db.generate_reading_score_histogram()
            >>> type(fig)
            <class 'matplotlib.figure.Figure'>
        """
        fig = plt.figure()
        # >>> YOUR CODE HERE >>>
        axes = fig.gca()

        self.df['ReadingScore'].hist(bins=15, edgecolor='black', ax=axes)

        axes.set_title('Distribution')
        axes.set_xlabel('Score')
        axes.set_ylabel('Freq')

        # <<< END OF YOUR CODE <<<


        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(
            __file__), "reading_score_histogram.png"))

        return fig

    def generate_math_writing_scatterplot(self) -> plt.Figure:
        """
        This function generates a scatterplot of the math and writing scores
        with appropriate labels and title. The edge color of the point is set
        to white and transparency to 0.95 for the scatterplot. It also adds a
        regression line to the scatterplot. The scatterplot is saved as a png
        file and returned as a matplotlib Figure object.

        Parameters:
            None

        Returns:
            plt.Figure: The scatterplot of the math and writing scores.

        Examples:
            >>> db = DataBasics(os.path.join(os.path.dirname(__file__), \
                "exam_original.csv"))
            >>> fig = db.generate_math_writing_scatterplot()
            >>> type(fig)
            <class 'matplotlib.figure.Figure'>
        """
        fig = plt.figure()
        # >>> YOUR CODE HERE >>>
        axes = fig.gca()

        axes.scatter(self.df['MathScore'], self.df['WritingScore'], alpha=0.5, edgecolors='none')
        axes.set_title('Math vs Writing')
        axes.set_xlabel('Math')
        axes.set_ylabel('Writing')
        
        # <<< END OF YOUR CODE <<<


        fig.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(
            __file__), "math_writing_scatterplot.png"))

        return fig


if __name__ == "__main__":
    import doctest
    import os

    from utils import print_green, print_red

    # Run the doctests. If all tests pass, print "All tests passed!"
    # You may ignore PYDEV DEBUGGER WARNINGS that appear in the console.
    if doctest.testmod(optionflags=doctest.ELLIPSIS).failed == 0:
        print_green("\nAll tests passed!\n")
    else:
        print_red("\nSome tests failed!\n")
