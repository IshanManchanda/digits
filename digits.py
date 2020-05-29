import os

import wandb

from src.gui import run_gui
from src.network import NeuralNetwork
from src.utils import deskew_data, load_data
from train import train


def main():
	if not os.path.isfile('data/mnist_py3_deskewed.pkl.gz'):
		x = input('Deskewed data not found, generate now? (y/n): ')
		if x.lower() == 'y':
			deskew_data()

	if os.path.isfile('networks/network.json'):
		x = input('Trained network found, train new network anyways? (y/n: ')
		if x.lower() == 'y':
			# TODO: Backup current network and train new one
			old_path = ''
			new_path = ''
			os.rename(old_path, new_path)
			pass
		else:
			gui()

	# TODO: Refactor these functions to make sense
	training, validation, test = load_data()
	wandb.init(project='digits')
	n = train([784, 128, 10], 0.008, 0.2, 0.05, training, validation)
	try:
		run_gui(n)
	except:
		wandb.run.save()


def gui():
	# TODO: Get path to 'current' network.json
	path = ''
	n = NeuralNetwork.load(path)
	run_gui(n)


if __name__ == '__main__':
	os.chdir(os.path.dirname(os.path.abspath(__file__)))
	main()
