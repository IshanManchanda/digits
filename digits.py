import os

import numpy as np
import wandb

from src.network import NeuralNetwork
from src.utils import deskew_data, load_data


def main():
	if not os.path.isfile('data/mnist_py3_deskewed.pkl.gz'):
		x = input('Deskewed data not found, generate now? (y/n): ')
		if x.lower() == 'y':
			deskew_data()

	training, validation, test = load_data()
	train([784, 128, 10], 0.008, 0, 0.05, training, validation)


# root = tk.Tk()
# InputGUI(root, n)
# root.mainloop()

def train(size, eta, lmbda, alpha, training, validation):
	wandb.init(project='digits')
	n = NeuralNetwork(size, eta=eta, lmbda=lmbda, alpha=alpha)
	n.train(
		np.random.permutation(training)[:5000],
		np.random.permutation(validation)[:500], epochs=20, batch_size=20
	)

	data_dir = os.path.join(os.getcwd(), 'networks')
	i = get_save_index(data_dir)
	n.save(os.path.join(data_dir, f'{i}.json'))
	n.plot(os.path.join(data_dir, f'{i}.png'))
	wandb.save(os.path.join(data_dir, f'{i}.json'))
	wandb.save(os.path.join(data_dir, f'{i}.png'))
	wandb.run.save()


def get_save_index(data_dir):
	i = 1
	while os.path.isfile(os.path.join(data_dir, f'{i}.json')):
		i += 1
	return i


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	main()
