import os

import numpy as np
import wandb

from network import NeuralNetwork


def train(size, eta, lmbda, alpha, training, validation):
	n = NeuralNetwork(size, eta=eta, lmbda=lmbda, alpha=alpha)
	n.train(
		np.random.permutation(training)[:5000],
		np.random.permutation(validation)[:500], epochs=5, batch_size=20
	)

	data_dir = os.path.join(os.getcwd(), 'networks')
	i = get_save_index(data_dir)
	data_dir = os.path.join(data_dir, str(i))
	# TODO: Save to run folder
	#  os.path.join(wandb.run.dir, '')
	n.save(os.path.join(data_dir, 'network.json'))
	n.plot(os.path.join(data_dir, 'accuracy.png'))
	wandb.save(os.path.join(data_dir, 'network.json'))
	wandb.save(os.path.join(data_dir, 'accuracy.png'))
	return n


def get_save_index(data_dir):
	i = 1
	while os.path.isdir(os.path.join(data_dir, str(i))):
		i += 1
	return i


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
