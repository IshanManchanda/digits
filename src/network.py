import numpy as np


# TODO: Change derivative calculation for last layer
# TODO: Check the derivatives for all layers
# TODO: Add a method that computes the cost on the validation set, print this
#  cost every epoch
# TODO: Graph cost vs epoch for various hyper parameters.
# TODO: Store all trained models
# TODO: Stop training if drop in cost is less than some epsilon (prolonged)
#  use a "increasing patience level" method to change epsilon.
# TODO: Add a test suite which will test against the test data.
#  it should firstly see how many predictions are correct, and also compute
#  the total cost in prediction.


class LeakyReLU:
	# The Leaky Rectifier function will serve as our activation function
	def __init__(self, alpha):
		self.alpha = alpha  # Parameter for the function

	def f(self, z):
		# The Leaky Rectifier is computed as:
		# f(z) = {        z,   z >= 0
		#        {alpha * z,   z < 0
		return np.where(z >= 0, z, self.alpha * z)

	def d(self, a):
		# The derivative of the function is given by:
		# f'(z) = {    1,   z >= 0 which implies f(z) >= 0
		#         {alpha,    z < 0 which implies f(z) < 0
		return np.where(a >= 0, 1, self.alpha)


class NeuralNetwork:
	def __init__(self, ns, eta=0.01, lmbda=0.1, alpha=0.01, ):
		# Network Structure
		self.l = len(ns)  # Number of layers
		self.ns = ns  # Number of neurons in each layer

		# Hyper Parameters
		self.eta = eta  # Learning rate
		self.lmbda = lmbda  # Coefficient of Regularization
		self.act_fn = LeakyReLU(alpha)  # Parameter for LReLU

		# Randomly initialize thetas (weights) with a normal distribution
		# of mean zero and variance as the reciprocal of the number of inputs
		# received by the particular neuron.
		self.thetas = [
			np.random.randn(ns[i], ns[i - 1]) / np.sqrt(ns[i - 1])
			for i in range(1, self.l)
		]

		# Similarly, initialize the biases with a distribution of mean zero
		# and std deviation and/or variance = 1
		self.biases = [np.random.randn(x, 1) for x in ns[1:]]

	def predict(self, x):
		# Our prediction is simply the activations of the output (last) layer.
		return self.get_activations(x)[-1]

	def train(self, xs, ys, epochs=10, batch_size=20):
		# We generate all the indices for the training data.
		# We will shuffle these indices each epoch to randomly order the data.
		# This is more efficient than zipping and shuffling the arrays.
		perm = np.arange(len(xs))

		for i in range(epochs):
			np.random.shuffle(perm)

			# We split the training data in batches, each of size batch_size.
			for j in range(0, len(xs), batch_size):
				batch = list(zip(xs[j:j + batch_size], ys[j:j + batch_size]))

				# Each batch is then used to train the network
				self.train_batch(batch)

			# After each epoch, optionally print progress
			print(f'Epoch {i}: ')

	def get_activations(self, x):
		# Validate the size of the input.
		self.validate_input(x)

		# We scale down the input to be in [0, 1].
		# Also, we add a 1 to the end to account for the bias.
		activations = [np.append(x / 255, 1)]

		# Iterate over each layer, excluding the output layer
		for theta, bias in zip(self.thetas[:-1], self.biases[:-1]):
			# The input to this layer is the matrix product of the weights
			# associated with this layer and the activations of the previous
			# layer plus the biases.
			z = np.array(np.dot(theta, activations[-1]) + bias,
				dtype='float64')

			# Apply the activation (LReLU) function to get the activations
			# for this layer, and add a 1 to the end for the bias.
			activations.append(self.act_fn.f(z))

		# For the output layer, we apply the softmax function, computed as:
		# exp(z) / sum(exp(z in zs))
		# The sum of outputs is clearly 1, which gives us
		# a useful interpretation as the 'confidence' for each possible output.
		z = np.dot(self.thetas[-1], activations[-1]) + self.biases[-1]
		activations.append(softmax(z))
		return activations

	def train_batch(self, batch):
		delta_thetas = [
			np.zeros((self.ns[i], self.ns[i - 1]))
			for i in range(1, self.l)
		]
		delta_biases = [np.zeros((x, 1)) for x in self.ns[1:]]

		# Iterate over all training sets
		for x, y in batch:
			# Activations for the current training set
			activations = self.get_activations(x)

			# We can trivially compute the bias and weight derivatives
			# for the last layer with the help of y and the prediction.
			# These formulae are derived by applying the chain rule to
			# the softmax (activation) and the log loss (error) functions.
			difference = activations[-1] - y
			delta_biases[-1] = difference
			delta_thetas[-1] = np.dot(difference, activations[-1].transpose())

			# The derivatives array stores the derivatives of the error function
			# with respect to the activations.
			# This array is used for backpropagation.
			derivatives = [np.zeros((x, 1)) for x in self.ns[1:-1]]
			derivatives[-1] = np.dot(self.thetas[-1].transpose(), difference)

			# Loop over all hidden layers, backwards
			for i in range(self.l - 2, 0, -1):
				# Assign some variables to make the following code cleaner.
				a, e = activations[i], derivatives[i + 1]
				a_t = a.reshape((-1, 1)).transpose()
				theta_t = self.thetas[i].transpose()

				# We discard the error in the bias unit if it is present,
				# opting to only adjust its theta.
				# e = e[:-1] if theta_t.shape[1] != e.shape[0] else e

				# The error for the layer is the matrix product of the thetas
				# for that layer and the errors for the next layer, times
				# the derivative of the activation function: (a * (1 - a))
				# errors[i] = np.dot(theta_t, e) * self.derivative(a)
				# delta_thetas[i] += np.multiply(e.reshape((-1, 1)), a_t)

		# Delta, the change in the parameters as indicated by this set,
		# is given by the matrix product of the errors of the next
		# layer and the transpose of the current activations.
		# This value is added so as to 'accumulate' it over the entire
		# training database.
		# deltas[i] += np.multiply(e.reshape((-1, 1)), a_t)

		change = [d / len(batch) for d in delta_thetas]
		# print('Change shape: ', [a.shape for a in change])
		# print('Deltas shape: ', [d.shape for d in deltas])
		for i in range(self.l - 1):
			self.thetas[i] -= change[i]

	def validate_input(self, x):
		if len(x) != self.ns[0]:
			raise ValueError(
				'Number of inputs != number of input neurons! '
				f'Expected: {self.ns[0]}, received: {len(x)}'
			)


def log_loss(y, a):
	# Our chosen error function is the Log Loss function.
	# This is also called the "Cross Entropy Loss" and is computed as:
	# f(a) = -y * log(a) - (1 - y) * log(1 - a)
	# i.e, -log(a) if y is 1, and -log(1 - a) if y is 0.
	return np.where(y, -np.log(a), -np.log(1 - a))


def softmax(z):
	# We offset each element of z by the maximum to prevent overflows.
	# nan_to_num is used to handle extremely small values. These are
	# made 0.
	z = np.nan_to_num(np.exp(z - np.max(z)))
	z_sum = np.sum(z)
	return np.nan_to_num(z / z_sum)


def main():
	n = NeuralNetwork([784, 1024, 10])
	# test_one(n, 0, examples=10)
	# train_all(n, examples=1000, cycles=5)
	# test_all(n, examples=20)
	train_all(n, examples=100, cycles=50)
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
		errors[j] = np.sum((y - predictions[j]) ** 2)

	print(f"\nRan tests for digit {i}, {examples} examples.")
	print("Average prediction:", predictions.mean(0))
	print("Average squared mean error:", errors.mean(), '\n')


def test_all(n, examples=1):
	for i in range(10):
		test_one(n, i, examples)


main()


# def train_batch(self, batch):
# 	delta_thetas = [
# 		np.zeros((self.ns[i], self.ns[i - 1]))
# 		for i in range(1, self.l)
# 	]
# 	delta_biases = [np.zeros((x, 1)) for x in self.ns[1:]]
#
# 	# Iterate over all training sets
# 	for x, y in batch:
# 		# Activations for the current training set
# 		activations = self.get_activations(x)
#
# 		# The error in the prediction is simply the difference
# 		# between the predicted and actual outputs.
# 		errors = [np.zeros((x, 1)) for x in self.ns[1:]]
# 		errors[-1] = log_loss(activations[-1], y)
# 		difference = activations[-1] - y
# 		delta_thetas[-1] = difference * activations[-1].transpose()
# 		delta_biases[-1] = difference
# 		# output_derivative =
# 		# errors[-2] = np.dot(self.thetas[-1].transpose(), errors[-1]) *
# 		# self.derivative(a)
#
# 		# Loop over all hidden layers, backwards
# 		for i in range(self.l - 2, 0, -1):
# 			# Assign some variables to make the following code cleaner.
# 			a, e = activations[i].copy(), errors[i + 1].copy()
# 			a_t = a.reshape((-1, 1)).transpose()
# 			theta_t = self.thetas[i].transpose()
#
# 			# We discard the error in the bias unit if it is present,
# 			# opting to only adjust its theta.
# 			e = e[:-1] if theta_t.shape[1] != e.shape[0] else e
#
# 			# The error for the layer is the matrix product of the thetas
# 			# for that layer and the errors for the next layer, times
# 			# the derivative of the activation function: (a * (1 - a))
# 			errors[i] = np.dot(theta_t, e) * self.derivative(a)
# 			delta_thetas[i] += np.multiply(e.reshape((-1, 1)), a_t)
#
# 	# Delta, the change in the parameters as indicated by this set,
# 	# is given by the matrix product of the errors of the next
# 	# layer and the transpose of the current activations.
# 	# This value is added so as to 'accumulate' it over the entire
# 	# training database.
# 	# deltas[i] += np.multiply(e.reshape((-1, 1)), a_t)
#
# 	change = [d / len(batch) for d in delta_thetas]
# 	# print('Change shape: ', [a.shape for a in change])
# 	# print('Deltas shape: ', [d.shape for d in deltas])
# 	for i in range(self.l - 1):
# 		self.thetas[i] -= change[i]
