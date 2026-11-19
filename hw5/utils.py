import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision


def print_red(msg):
    print("\033[91m {}\033[00m" .format(msg))

def print_green(msg):
    print("\033[92m {}\033[00m" .format(msg))

def print_yellow(msg):
    print("\033[93m {}\033[00m" .format(msg))

def print_blue(msg):
    print("\033[94m {}\033[00m" .format(msg))
    
def assert_less_equal(actual, expected, msg):
    if actual < expected:
        print_green(msg)
    else:
        print_red(msg)

def assert_greater_equal(actual, expected, msg):
    if actual > expected:
        print_green(msg)
    else:
        print_red(msg)

def plot_explained_variance_ratio(explained_variance_ratio, filename: str = 'explained_variance_ratio.png') -> None:
    """
    Plot explained variance ratio of the data. The x-axis should be the number
    of components and the y-axis should be the cumulative explained variance 
    ratio. Save the plot to a file given by filename.

    Args:
        filename: The filename to save the plot to.

    Returns:
        None.
    """

    plt.plot(explained_variance_ratio,
             label='Explained Variance Ratio')

    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()

    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


def compare_digit_images(filename, X, X_pca):
    fig, axs = plt.subplots(2, 5, figsize=(1.5*5, 2*2))

    for i in range(5):
        img = X[i].reshape((32, 32))
        axs[0, i].imshow(img, cmap='gray')

    for i in range(5):
        img = X_pca[i].reshape((32, 32))
        axs[1, i].imshow(img, cmap='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), filename))


def load_imagenette():
    # images vary in size (shorter side is 160px), so convert to
    # (32, 32) grayscale by resizing the shorter side and center cropping
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(32),
        torchvision.transforms.CenterCrop(32),
    ])
    imagenette_train = torchvision.datasets.Imagenette(root='./data', split='train', size='160px', download=True, transform=transform)
    imagenette_test = torchvision.datasets.Imagenette(root='./data', split='val', size='160px', download=True, transform=transform)

    X_train, y_train = zip(*imagenette_train)
    X_test, y_test = zip(*imagenette_test)

    # these load as lists of (32, 32) grayscale PIL images
    X_train = np.array([np.asarray(img) for img in X_train], dtype=np.float64)
    X_test  = np.array([np.asarray(img) for img in X_test],  dtype=np.float64)

    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    return X_train, y_train, X_test, y_test
