import gzip
import os
import pickle

import numpy as np
from PIL import Image

from src.globals import deskew_path, mini_path, mnist_path
from src.preprocessor import deskew_image


def load_data(mini=True, deskew=True):
    """
    Loads MNIST dataset from disk.

    Returns the training, validation, and test data as a list of tuples,
    where the outputs are one-hot encoded vectors.
    """
    # TODO: Optimize memory usage by loading only parts.
    file_path = mini_path if mini else deskew_path if deskew else mnist_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'{file_path} not found!')

    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    processed_data = []

    # Data contains 3 "sections": training, validation, and test
    # containing 50K, 10K, and 10K examples each (full; mini is 1/10 of this).
    # We return this data as a list of tuples. First element in each tuple
    # is a numpy x matrix and the second is a numpy y matrix.
    for section in data:
        # X matrix is 784 x n
        x = np.array(section[0]).T
        # One-hot encode the y values to get a 10 x n matrix.
        y = one_hot_encode(section[1])
        processed_data.append((x, y))
    return processed_data


def deskew_data():
    """
    Deskews the MNIST dataset and saves it to disk.
    """
    # Check if deskewed data already exists
    exist_deskew = os.path.isfile(deskew_path)
    exist_mini = os.path.isfile(mini_path)
    if exist_deskew and exist_mini:
        return

    with gzip.open(mnist_path, 'rb') as f:
        data = pickle.load(f)

    # TODO: Selectively process data if only mini needs to be generated.
    processed_data = []
    for section in data:
        xs = []
        for x in section[0]:
            xs.append(list(deskew_image(x.reshape((28, 28))).reshape(784, )))
        processed_data.append((xs, section[1]))

    # TODO: Package into parts of 10k/5k/5k.
    if not exist_mini:
        processed_data_mini = [
            (processed_data[0][0][:5000], processed_data[0][1][:5000]),
            (processed_data[0][0][:1000], processed_data[0][1][:1000]),
            (processed_data[0][0][:1000], processed_data[0][1][:1000])
        ]
        with gzip.open(mini_path, 'wb') as f:
            # A protocol of -1 means the latest one
            pickle.dump(processed_data_mini, f, protocol=-1)

    if not exist_deskew:
        with gzip.open(deskew_path, 'wb') as f:
            # A protocol of -1 means the latest one
            pickle.dump(processed_data, f, protocol=-1)


def one_hot_encode(digits):
    """
    Returns a one-hot encoded vector matrix of the input digit list.
    """
    y = np.zeros((10, len(digits)))
    for i, d in enumerate(digits):
        y[d, i] = 1
    return y


def draw_digit(image):
    """
    Renders and displays the image.
    """
    arr = (np.array(image).reshape((28, 28)) * 255).astype('uint8')
    Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).show()


def save_digit(image, file_path):
    """
    Renders and saves the image to disk.
    """
    arr = (np.array(image).reshape((28, 28)) * 255).astype('uint8')
    Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).save(file_path)


def main():
    deskew_data()


if __name__ == '__main__':
    main()
