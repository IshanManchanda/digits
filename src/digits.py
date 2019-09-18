import os
import tkinter as tk

import numpy as np

from gui import InputGUI
from network import NeuralNetwork
from utils import deskew_data, load_data


def main():
	if not os.path.isfile('../data/mnist_py3_deskewed.pkl.gz'):
		x = input('Deskewed data not found, generate now? (y/n): ')
		if x.lower() == 'y':
			deskew_data()

	training, validation, test = load_data()
	train([784, 128, 10], 0.008, 0, 0.05, training, validation)
	train([784, 128, 10], 0.008, 0, 0.05, training, validation)
	train([784, 128, 10], 0.008, 0, 0.05, training, validation)
	train([784, 128, 10], 0.0075, 0, 0.05, training, validation)
	train([784, 128, 10], 0.0075, 0, 0.05, training, validation)
	train([784, 128, 10], 0.0075, 0, 0.05, training, validation)


# root = tk.Tk()
# InputGUI(root, n)
# root.mainloop()

def train(size, eta, lmbda, alpha, training, validation):
	n = NeuralNetwork(size, eta=eta, lmbda=lmbda, alpha=alpha)
	n.train(
		np.random.permutation(training)[:5000],
		np.random.permutation(validation)[:500], epochs=20, batch_size=20
	)
	i = 1
	while os.path.isfile(f'..\\networks\\{i}.json'):
		i += 1
	n.save(f'..\\networks\\{i}.json')
	n.plot(f'..\\networks\\{i}.png')


if __name__ == '__main__':
	main()
