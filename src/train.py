import os

import numpy as np
import wandb

from globals import current_dir
from network import NeuralNetwork


def train(size, eta, lmbda, alpha, training, validation):
	n = NeuralNetwork(size, eta=eta, lmbda=lmbda, alpha=alpha)
	n.train(
		np.random.permutation(training)[:5000],
		np.random.permutation(validation)[:500], epochs=5, batch_size=20
	)

	network_path = os.path.join(current_dir, 'network.json')
	n.save(network_path)
	wandb.save(network_path)

	n.plot(os.path.join(current_dir, 'accuracy.png'))
	return n
