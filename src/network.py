import numpy as np


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
		# Note: The coefficient is called "lmbda" as "lambda" is a keyword.

		# Activation Function Object
		self.act_fn = LeakyReLU(alpha)  # Parameter for LReLU

		# Randomly initialize thetas (weights) with a normal distribution
		# of mean zero and variance as the reciprocal of the number of inputs
		# received by the particular neuron.
		self.thetas = [
			np.random.randn(ns[i], ns[i - 1]) / np.sqrt(ns[i - 1])
			for i in range(1, self.l)
		]

		# Similarly, initialize the biases with a distribution of mean zero
		# and standard deviation and variance 1
		self.biases = [np.random.randn(x) for x in ns[1:]]

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
			# print(f'Epoch {i}: ')

	def get_activations(self, x):
		# Validate the size of the input.
		self.validate_input(x)

		# We scale down the input to be in [0, 1].
		# Also, we add a 1 to the end to account for the bias.
		activations = [x / 255]

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
		delta_biases = [np.zeros((x,)) for x in self.ns[1:]]

		# Iterate over all examples in the batch
		for x, y in batch:
			# Activations for the current training example
			activations = self.get_activations(x)

			# We can trivially compute the bias and weight derivatives
			# for the last layer with the help of y and the prediction.
			# These formulae are derived by applying the chain rule to
			# the softmax (activation) and the log loss (error) functions.
			difference = np.array(activations[-1] - y)
			delta_biases[-1] += difference
			delta_thetas[-1] += np.dot(
				difference.reshape((-1, 1)),
				activations[-2].reshape(1, -1)
			)

			# The derivatives array stores the derivatives of the error function
			# with respect to the activations.
			# This array is used for backpropagation.
			derivatives = [np.zeros((x,)) for x in self.ns]
			# Derivative of the second last layer
			# i.e, layer before output layer
			# Change to [-2] or in init make self.ns[1:-1] since
			# input and output layers won't have derivatives.
			derivatives[-2] = np.dot(self.thetas[-1].transpose(), difference)

			# Loop over all hidden layers, backwards
			for i in range(self.l - 3, -1, -1):
				# Assign some variables to make the following code cleaner.
				a_t = activations[i].reshape((1, -1))
				theta_t = self.thetas[i].transpose()

				# We compute a term called the "error".
				# This can be considered similar to the "difference" term above
				# with the derivative of the activation function multiplied.
				error = derivatives[i + 1] * self.act_fn.d(activations[i + 1])
				e1 = error.reshape((-1, 1))

				# The following formulae used to compute the derivatives are
				# derived using the chain rule.
				# Our goal is to compute the derivatives of the error function
				# with respect to each parameter, to tweak them as necessary.
				# These values are added so as to 'accumulate' them
				# over the batch we are currently training on.
				delta_biases[i] += error
				delta_thetas[i] += np.dot(e1, a_t)
				derivatives[i] = np.dot(theta_t, error)

		for i in range(self.l - 1):
			self.thetas[i] -= delta_thetas[i] / len(batch)
			self.biases[i] -= delta_biases[i] / len(batch)

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
	n = NeuralNetwork([784, 16, 10])
	# test_one(n, 0, examples=10)
	# train_all(n, examples=1000, cycles=5)
	# test_all(n, examples=20)
	train_all(n, examples=100, cycles=50)
	test_all(n, examples=20)


def get_images(digit, number):
	with open(f'..\\data\\temp_data\\data{digit}', 'rb') as f:
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
