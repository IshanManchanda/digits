import gzip
import os
import pickle

import numpy as np
from PIL import Image

from preprocessor import deskew_image


def load_data(mini=True, deskew=True):
	"""
	Loads MNIST dataset from disk.

	Returns the training, validation, and test data as a list of tuples,
	where the outputs are one-hot encoded vectors.
	"""
	# TODO: Optimize memory usage by loading only parts.

	# The _py3 version of the dataset is a redumped version for Python 3
	# which doesn't use Python 2's latin1 encoding
	data_dir = os.path.join(os.getcwd(), 'data')
	file_path = (
		os.path.join(data_dir, 'mnist_py3_mini_deskewed.pkl.gz') if mini
		else os.path.join(data_dir, 'mnist_py3_deskewed.pkl.gz') if deskew
		else os.path.join(data_dir, 'mnist_py3.pkl.gz')
	)

	if not os.path.isfile(file_path):
		raise FileNotFoundError(f'{file_path} not found!')

	with gzip.open(file_path, 'rb') as f:
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
	data_dir = os.path.join(os.getcwd(), 'data')
	deskew_path = os.path.join(data_dir, 'mnist_py3_deskewed.pkl.gz')
	if os.path.isfile(deskew_path):
		return

	data_path = os.path.join(data_dir, 'mnist_py3.pkl.gz')
	with gzip.open(data_path, 'rb') as f:
		data = pickle.load(f)

	processed_data = []
	for section in data:
		xs = []
		for x in section[0]:
			xs.append(list(deskew_image(x.reshape((28, 28))).reshape(784, )))
		processed_data.append((xs, section[1]))

	# TODO: Package into parts of 10k/5k/5k.

	# deskew_mini_path = os.path.join(data_dir,
	# 'mnist_py3_mini_deskewed.pkl.gz')
	# processed_data_mini = [
	# 	(processed_data[0][0][:5000], processed_data[0][1][:5000]),
	# 	(processed_data[0][0][:1000], processed_data[0][1][:1000]),
	# 	(processed_data[0][0][:1000], processed_data[0][1][:1000])
	# ]
	# with gzip.open(deskew_mini_path, 'wb') as f:
	# A protocol of -1 means the latest one
	# pickle.dump(processed_data_mini, f, protocol=-1)

	with gzip.open(deskew_path, 'wb') as f:
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


def save_digit(image, file_path):
	"""
	Renders and saves the image to disk.
	"""
	arr = (np.array(image).reshape((28, 28)) * 255).astype('uint8')
	Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).save(file_path)


def main():
	deskew_data()


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	main()
