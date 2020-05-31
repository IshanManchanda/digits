import os

import wandb

from globals import archive_dir, current_dir
from gui import run_gui
from network import NeuralNetwork
from train import train
from utils import deskew_data, load_data


def main():
	check_deskewed_files_exist()

	if os.path.isfile('networks/network.json'):
		x = input('Trained network found, train new network anyways? (y/n: ')
		if x.lower() != 'y':
			gui()
		else:
			# TODO: Generate name from timestamp
			new_dir = os.path.join(archive_dir, '1')
			os.rename(current_dir, new_dir)

	# REVIEW: Pass the loaded data as a global
	#  so that it can be used across runs?
	training, validation, test = load_data()
	wandb.init(project='digits')
	n = train([784, 128, 10], 0.008, 0.2, 0.05, training, validation)
	try:
		run_gui(n)
	except:
		wandb.run.save()


def gui():
	network_path = os.path.join(current_dir, 'network.json')
	n = NeuralNetwork.load(network_path)
	run_gui(n)


def check_deskewed_files_exist():
	# TODO: Check for all parts of deskewed dataset
	if not os.path.isfile('data/mnist_py3_deskewed.pkl.gz'):
		x = input('Deskewed data not found, generate now? (y/n): ')
		if x.lower() == 'y':
			deskew_data()


if __name__ == '__main__':
	main()
