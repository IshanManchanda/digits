import os

import wandb

from src.globals import current_dir, network_path
from src.network import NeuralNetwork


def train(size, alpha, lmbda, beta, epochs, batch_size, training, validation):
    nn = NeuralNetwork(size, alpha=alpha, lmbda=lmbda, beta=beta)
    nn.train(training, validation, epochs=epochs, batch_size=batch_size)

    nn.save(network_path)
    wandb.save(network_path)

    nn.plot(os.path.join(current_dir, 'accuracy.png'))
    return nn
