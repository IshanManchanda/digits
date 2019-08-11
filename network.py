import numpy as np


# TODO: Fix error on line 76:
#  ValueError: shapes (17,16) and (17,) not aligned: 16 (dim 1) != 17 (dim 0)

# TODO: Switch to Leaky ReLU or something

class NeuralNetwork:
	def __init__(self, l, ns):
		self.l = l  # Number of layers
		self.ns = ns  # Number of neurons in each layer

		# Randomly initialize thetas and biases
		self.thetas = [
			(np.random.rand(ns[i], ns[i - 1] + 1) * 2) - 1
			for i in range(1, l)
		]

	def predict(self, x):
		# Our prediction is simply the activations of the output (last) layer.
		return self.get_activations(x)[-1]

	def get_activations(self, x):
		# Validate the size of the input.
		if len(x) != self.ns[0]:
			raise ValueError(
				'Number of inputs != number of input neurons! '
				f'Expected: {self.ns[0]}, received: {len(x)}'
			)

		# We add a 1 to the end to account for the bias.
		activations = [np.append(x, 1)]

		# Iterate over each layer, excluding the output layer
		for i in range(self.l - 2):
			# The input to this layer is the matrix product of the weights
			# associated with this layer and the activations of the previous
			# layer.
			z = np.dot(self.thetas[i], activations[i])

			# Apply the activation (sigmoid) function to get the activations
			# for this layer, and add a 1 to the end for the bias.
			activations.append(np.append(1 / (1 + np.exp(-z)), 1))

		# For the output layer, we apply the softmax function, computed as:
		# exp(z) / sum(exp(z in zs))
		# The sum of outputs is clearly 1, which gives us
		# a useful interpretation as the 'confidence' for each possible output.
		z = np.exp(np.dot(self.thetas[-1], activations[-1]))
		z_sum = sum(z)
		activations.append(z / z_sum)

		return activations

	def train(self, xs, ys):
		deltas = [
			np.zeros((self.ns[i], self.ns[i - 1] + 1))
			for i in range(1, self.l)
		]

		# Iterate over all training sets
		for x, y in zip(xs, ys):
			# Activations for the current training set
			activations = self.get_activations(x)

			# The error in the prediction is simply the difference
			errors = [np.array([0]) for _ in range(self.l)]
			errors[-1] = activations[-1] - y

			# Loop over all hidden layers, backwards
			for i in range(self.l - 2, 0, -1):
				# Assign some variables to make the following code cleaner.
				theta_t = self.thetas[i].transpose()
				a = activations[i]
				a_t = a.reshape((-1, 1)).transpose()

				# The error for the layer is the matrix product of the thetas
				# for that layer and the errors for the next layer, times
				# the derivative of the activation function (a * (1 - a))
				errors[i] = np.dot(theta_t, errors[i + 1]) * a * (1 - a)

				# Delta, the change in the parameters as indicated by this set,
				# is given by the matrix product of the errors of the next
				# layer and the transpose of the current activations.
				# This value is added so as to 'accumulate' it over the entire
				# training database.
				deltas[i] += np.multiply(errors[i + 1][:, np.newaxis], a_t)

		# noinspection PyTypeChecker
		change = [d / len(xs) for d in deltas]
		# print('Change shape: ', [a.shape for a in change])
		# print('Deltas shape: ', [d.shape for d in deltas])
		for i in range(self.l - 1):
			self.thetas[i] -= change[i]


def main():
	n = NeuralNetwork(4, [784, 16, 16, 10])
	train_all(n, examples=100)
	test_all(n, examples=20)


# test_one(n, i=0, examples=10)

# train_one(n, i=0, examples=1000, cycles=1)
# test_one(n, i=0, examples=10)

# train_one(n, i=1, examples=1000)
# test_one(n, i=0, examples=10)
# test_one(n, i=1, examples=10)

# train_one(n, i=0, examples=100, cycles=1)
# test_one(n, i=0, examples=10)
# test_one(n, i=1, examples=10)


#
# num = 1000
# ims0 = ims.copy()
# with open('training_data\\data1', 'rb') as f:
# 	ims = [int(x) for x in f.read(28 * 28 * num)]
#
# ims = np.array(ims).reshape((num, 28 * 28))
# y0 = y.copy()
# y = np.array([0, 1] + [0] * 8)
#
# before = after.copy()
# for i in range(1):
# 	n.train(ims, [y] * num)
# after = n.predict(im)[-1]
#
# print()
# print('Prediction before training:', before)
# print('Prediction after training: ', after)
# print()
# print('Offset before training: ', y0 - before)
# print('Offset after training: ', y0 - after)
# print()
# print('Squared mean error before: ', sum((y0 - before) ** 2))
# print('Squared mean error after: ', sum((y0 - after) ** 2))
#
# n.train(ims0, [y0] * num)
# after = n.predict(im)[-1]
#
# print()
# print('Prediction before training:', before)
# print('Prediction after training: ', after)
# print()
# print('Offset before training: ', y0 - before)
# print('Offset after training: ', y0 - after)
# print()
# print('Squared mean error before: ', sum((y0 - before) ** 2))
# print('Squared mean error after: ', sum((y0 - after) ** 2))


def get_images(digit, number):
	with open(f'training_data\\data{digit}', 'rb') as f:
		ims = [int(x) for x in f.read(28 * 28 * number)]
	return np.array(ims).reshape((number, 28 * 28))


def get_expected_y(digit):
	y = np.array([0] * 10)
	y[digit] = 1
	return y


def train_one(n, i, examples=1000, cycles=1):
	ims = get_images(i, examples)
	y = get_expected_y(i)

	for j in range(cycles):
		n.train(ims, [y] * examples)


def train_all(n, examples=1000, cycles=1):
	ims, ys = [], []

	for i in range(10):
		ims.extend(get_images(i, examples))
		ys.extend([get_expected_y(i)] * examples)

	for i in range(cycles):
		n.train(np.array(ims), np.array(ys))


def test_one(n, i, examples=1):
	ims = get_images(i, examples)
	y = get_expected_y(i)

	errors = np.zeros((examples,))
	predictions = np.zeros((examples, 10))

	for j in range(examples):
		predictions[j] = n.predict(ims[j])
		errors[j] = sum((y - predictions[j]) ** 2)

	print(f"\nRan tests for digit {i}, {examples} examples.")
	print("Average prediction:", predictions.mean(0))
	print("Average squared mean error:", errors.mean(), '\n')


def test_all(n, examples=1):
	for i in range(10):
		test_one(n, i, examples)


main()
