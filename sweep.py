import numpy as np
import wandb

from src.network import NeuralNetwork
from src.utils import load_data

# Set up your default hyperparameters
hyperparameter_defaults = dict(
	n_layers=1,
	n_neurons=128,
	eta=0.008,
	lmbda=0.2,
	alpha=0.05,
	epochs=20,
	batch_size=40,
	# size_training=5000,
	# size_validation=500,
)

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
# Access all hyperparameter values through wandb.config
config = wandb.config

training, validation, test = load_data()
# architecture = [784, 128, 10]
architecture = [784] + [config['n_neurons']] * config['n_layers'] + [10]

nn = NeuralNetwork(
	architecture, eta=config['eta'], lmbda=config['lmbda'],
	alpha=config['alpha']
)
nn.train(
	np.random.permutation(training)[:5000],
	np.random.permutation(validation)[:500],
	epochs=config['epochs'], batch_size=config['batch_size']
)
