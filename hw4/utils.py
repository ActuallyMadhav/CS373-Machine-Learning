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
    if actual <= expected:
        print_green(msg)
    else:
        print_red(msg)

def assert_greater_equal(actual, expected, msg):
    if actual >= expected:
        print_green(msg)
    else:
        print_red(msg)

def _load(dataset_cls, return_tensor=False):
    train = dataset_cls(root='./data', train=True, download=True)
    test = dataset_cls(root='./data', train=False, download=True)

    if return_tensor:
        return train.data, train.targets, test.data, test.targets
    return train.data.numpy(), train.targets.numpy(), test.data.numpy(), test.targets.numpy()

def load_mnist(return_tensor=False):
    """MNIST: 28x28 grayscale images of handwritten digits, 10 classes."""
    return _load(torchvision.datasets.MNIST, return_tensor)
