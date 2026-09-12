import numpy as np
import pandas as pd

def print_red(msg):
    print("\033[91m {}\033[00m" .format(msg))

def print_green(msg):
    print("\033[92m {}\033[00m" .format(msg))

def print_yellow(msg):
    print("\033[93m {}\033[00m" .format(msg))

def read_hw1_data(filename):
	data = pd.read_csv(filename).fillna("NA").astype(str)
	y = data.iloc[:, 0].values
	X = data.iloc[:, 1:].values

	return X, y

