import tkinter as tk
import numpy as np
from gui import InputGUI
from network import NeuralNetwork, load_data
import os


def main():
	training, validation, test = load_data()
	n = NeuralNetwork([784, 64, 10], eta=0.5, lmbda=0.05, alpha=0.05)
	n.train(
		np.random.permutation(training)[:5000],
		np.random.permutation(validation)[:500], epochs=50, batch_size=20
	)
	i = 1
	while os.path.isfile(f'..\\networks\\{i}.json'):
		i += 1
	n.save(f'..\\networks\\{i}.json')
	n.plot(f'..\\networks\\{i}.png')

	root = tk.Tk()
	InputGUI(root, n)
	root.mainloop()


if __name__ == '__main__':
	main()
