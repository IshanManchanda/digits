import numpy as np


class NeuralNetwork:
	def __init__(self, l, ns):
		self.l = l  # Number of layers
		self.ns = ns  # Number of neurons in each layer

		# Randomly initialize thetas and biases
		self.thetas = [
			np.random.rand(ns[i], ns[i - 1]) * 2 - 1 for i in range(1, l)]
		# print(f'Theta shapes: {[x.shape for x in self.thetas]}')
		# print(f'Thetas: {self.thetas}')
		self.biases = [np.random.rand(ns[i]) * 2 - 1 for i in range(1, l)]

	def predict(self, x):
		if len(x) != self.ns[0]:
			raise ValueError(
				'Number of inputs != number of input neurons! '
				f'Expected: {self.ns[0]}, received: {len(x)}'
			)

		activations = [x]

		# Iterate over each layer
		for i in range(self.l - 1):
			# The input to this layer is the matrix product of the weights
			# associated with this layer and the activations of the previous
			# layer.
			z = np.dot(self.thetas[i], activations[i])

			# Apply the activation (sigmoid) function to get the activations
			# for this layer.
			activations.append(1 / (1 + np.exp(-z)))

		# Finally, the activations of the output layer is our prediction.
		return activations

	def train(self, xs, ys):
		deltas = [
			np.zeros((self.ns[i], self.ns[i - 1])) for i in range(1, self.l)]

		# Iterate over all training sets
		for x, y in zip(xs, ys):
			# Activations for the current training set
			activations = self.predict(x)

			# The error in the prediction is simply the difference
			errors = [np.array([0]) for x in range(self.l)]
			errors[-1] = activations[-1] - y

			# Loop over all hidden layers, backwards
			for i in range(self.l - 2, 0, -1):
				# Theta transpose and activations for the current hidden layer
				theta_t = self.thetas[i].transpose()
				a = activations[i]
				# a_t = a.transpose()
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
	num = 1000
	with open('training_data\\data0', 'rb') as f:
		ims = [int(x) for x in f.read(28 * 28 * num)]
	im = ims[:28 * 28]
	ims = np.array(ims).reshape((num, 28 * 28))
	y = np.array([1] + [0] * 9)

	n = NeuralNetwork(5, [784, 1024, 1024, 1024, 10])

	before = n.predict(im)[-1]
	for i in range(1):
		n.train(ims, [y] * num)
	after = n.predict(im)[-1]

	print()
	print('Prediction before training:', before)
	print('Prediction after training: ', after)
	print()
	print('Offset before training: ', y - before)
	print('Offset after training: ', y - after)
	print()
	print('Squared mean error before: ', sum((y - before) ** 2))
	print('Squared mean error after: ', sum((y - after) ** 2))


main()
