import gzip
import os
import pickle

import numpy as np
from PIL import Image

from .preprocessor import deskew_image


def load_data(deskew=True):
	"""
	Loads MNIST dataset from disk.

	Returns the training, validation, and test data as a list of tuples,
	where the outputs are one-hot encoded vectors.
	"""
	# The _py3 version of the dataset is a redumped version for Python 3
	# which doesn't use Python 2's latin1 encoding
	if deskew:
		with gzip.open('../data/mnist_py3_deskewed.pkl.gz', 'rb') as f:
			data = pickle.load(f)
	else:
		with gzip.open('../data/mnist_py3.pkl.gz', 'rb') as f:
			data = pickle.load(f)
	processed_data = []

	# Data contains 3 "sections": training, validation, and test
	# containing 50K, 10K, and 10K examples each.
	# We return this data as a list of tuples containing input-output pairs
	for section in data:
		# We reshape the inputs, and convert the outputs into
		# "one-hot encoded vectors" which is just the output vector.
		processed_data.append(list(zip(
			[np.array(x).reshape((784,)) for x in section[0]],
			[get_expected_y(y) for y in section[1]]
		)))
	return processed_data


def deskew_data():
	"""
	Deskews the MNIST dataset and saves it to disk.
	"""
	# Check if deskewed data already exists
	if os.path.isfile('../data/mnist_py3_deskewed.pkl.gz'):
		return

	# This method deskews all the images and saves it to disk
	with gzip.open('../data/mnist_py3.pkl.gz', 'rb') as f:
		data = pickle.load(f)
	processed_data = []

	for section in data:
		xs = [
			list(deskew_image(x.reshape((28, 28))).reshape(784, ))
			for x in section[0]
		]
		processed_data.append((xs, section[1]))

	with gzip.open('../data/mnist_py3_deskewed.pkl.gz', 'wb') as f:
		# A protocol of -1 means the latest one
		pickle.dump(processed_data, f, protocol=-1)


def get_expected_y(digit):
	"""
	Returns a one-hot encoded vector of the inputted digit.
	"""
	y = np.array([0] * 10)
	y[digit] = 1
	return y


def draw_digit(image):
	"""
	Renders and displays the image.
	"""
	arr = (np.array(image).reshape((28, 28)) * 255).astype('uint8')
	Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).show()


def main():
	deskew_data()


if __name__ == '__main__':
	main()
