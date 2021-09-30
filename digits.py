import os

import wandb

from globals import archive_dir, current_dir, deskew_path, mini_path, \
	network_path
from gui import run_gui
from network import NeuralNetwork
from train import train
from utils import deskew_data, load_data


def main():
	if os.path.isfile(network_path):
		x = input('Trained network found, train new network anyways? (y/n): ')
		if x.lower() != 'y':
			gui()

		else:
			# TODO: Save the data to current dir as well as a run dir in archive
			#  while saving initially
			#  so that current dir can just be overwritten.
			# Move data from current dir to archive
			new_dir = os.path.join(archive_dir, '1')
			os.rename(current_dir, new_dir)

	deskew = check_deskewed_files()
	mini = False
	if deskew:
		x = input('Use entire dataset for training? (y/n): ')
		mini = x.lower() != 'y'

	# REVIEW: Pass the loaded data as a global
	#  so that it can be used across runs?
	training, validation, test = load_data(mini=mini, deskew=deskew)

	wandb.init(project='digits', entity='ishanmanchanda')
	architecture = [784, 128, 10]
	eta = 0.008
	lmbda = 0.2
	alpha = 0.05
	wandb.conf

	# [Network structure], eta, lambda, alpha, training, validation
	n = train(architecture, eta, lmbda, alpha, training, validation)
	try:
		run_gui(n)
	except:
		wandb.run.save()


def gui():
	n = NeuralNetwork.load(network_path)
	run_gui(n)


def check_deskewed_files():
	# TODO: Check for all parts of deskewed dataset
	if not os.path.isfile(deskew_path) or not os.path.isfile(mini_path):
		x = input('Deskewed data not found, generate now? (y/n): ')
		if x.lower() != 'y':
			return False

		deskew_data()
	return True


if __name__ == '__main__':
	main()
