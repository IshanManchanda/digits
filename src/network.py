import json
import time

import numpy as np
import wandb
from matplotlib import pyplot as plt

from src.utils import draw_digit, load_data


class LeakyReLU:
    # The Leaky Rectifier function will serve as our activation function
    def __init__(self, alpha):
        self.alpha = alpha  # Parameter for the function

    def f(self, z):
        # The Leaky Rectifier is computed as:
        # f(z) = {        z,   z >= 0
        #        {alpha * z,   z < 0
        return np.where(z >= 0, z, self.alpha * z)

    def d(self, a):
        # The derivative of the function is given by:
        # f'(z) = {    1,   z >= 0 which implies f(z) >= 0
        #         {alpha,    z < 0 which implies f(z) < 0
        return np.where(a >= 0, 1, self.alpha)


def log_loss(y, a):
    # Our chosen error function is the Log Loss function.
    # This is also called the "Cross Entropy Loss" and is computed as:
    # f(a) = -y * log(a) - (1 - y) * log(1 - a)
    # i.e, -log(a) if y is 1, and -log(1 - a) if y is 0.
    return np.where(y, -np.log(a), -np.log(1 - a)).sum()


def softmax(z):
    # We offset each element of z by the maximum to prevent overflows.
    # nan_to_num is used to handle extremely small values. These are
    # made 0.
    z = np.nan_to_num(np.exp(z - np.max(z)))
    z_sum = np.sum(z)
    return np.nan_to_num(z / z_sum)


class NeuralNetwork:
    def __init__(self, ns, eta=0.5, lmbda=0, alpha=0.05):
        print(f'ns: {ns}, eta: {eta}, lambda: {lmbda}, alpha: {alpha}')
        # Network Structure
        self.n = len(ns)  # Number of layers
        self.ns = ns  # Number of neurons in each layer

        # Hyper Parameters
        self.eta = eta  # Learning rate
        self.lmbda = lmbda  # Coefficient of Regularization
        # Note: The coefficient is called "lmbda" as "lambda" is a keyword.

        # Activation Function Object
        self.act_fn = LeakyReLU(alpha)  # Parameter for LReLU

        # Log hyperparameters in wandb to analyze later.
        wandb.config.update({
            'architecture': ns, 'eta': eta, 'lambda': lmbda, 'alpha': alpha
        })

        # Randomly initialize thetas (weights) with a normal distribution
        # of mean zero and variance as the reciprocal of the number of inputs
        # received by the particular neuron.
        self.thetas = [
            np.random.randn(ns[i], ns[i - 1]) / np.sqrt(ns[i - 1])
            for i in range(1, self.n)
        ]

        # Similarly, initialize the biases with a distribution of mean zero
        # and standard deviation and variance 1
        self.biases = [np.random.randn(x) for x in ns[1:]]

        # We use this performance variable to keep a track of how
        # the network is performing. We will use it plot graph(s) between
        # the number of correct predictions on the validation data and epochs.
        self.performance = []

    def predict(self, x):
        # Our prediction is simply the activations of the output (last) layer.
        return self.get_activations(x)[-1]

    def train(self, data, validation_data=None, epochs=10, batch_size=20):
        # We generate all the indices for the training data.
        # We will shuffle these indices each epoch to randomly order the data.
        # This is more efficient than zipping and shuffling the arrays directly
        perm = np.arange(len(data))
        self.performance = []
        n_validation = len(validation_data)

        # Log hyperparameters in wandb to analyze later.
        wandb.config.update({'epochs': epochs, 'batch_size': batch_size})

        # Log initial accuracy and loss for train and validation sets
        self.report_metrics(data, 'train', 0)
        if validation_data is not None:
            self.report_metrics(validation_data, 'dev', 0)

        for i in range(1, epochs + 1):
            np.random.shuffle(perm)

            # We split the training data in batches, each of size batch_size.
            for j in range(0, len(data), batch_size):
                # From the shuffled indices, we select a range
                # and pick all the examples in that range.
                batch = [data[x] for x in perm[j:j + batch_size]]

                # Each batch is then used to train the network
                self.train_batch(batch)

            # Compute metrics for training data for learning curve
            # Log the data in wandb and also locally.
            self.report_metrics(data, 'train', i)
            if validation_data is not None:
                perf = self.report_metrics(validation_data, 'dev', i)
                # Log just accuracy for now
                self.performance.append(perf[0])

    def plot(self, filename=None):
        if not self.performance:
            return

        plt.figure()
        plt.plot(
            range(1, len(self.performance) + 1),
            self.performance, 'r'
        )
        plt.title(
            f'ns: {self.ns}, eta: {self.eta}, '
            f'lambda: {self.lmbda}, alpha: {self.act_fn.alpha}.')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Prediction Accuracy (%)')
        if filename:
            plt.tight_layout()
            plt.savefig(filename)
        plt.show()

    def save(self, filename):
        # This function will save all parameters of our network.
        # We use this elaborate setup instead of simply pickling and dumping
        # the network object so that we can still use this data,
        # even if we change the architecture of our class.
        # Unpickling and loading will not work in that case.
        data = {
            'ns': self.ns,
            'eta': self.eta,
            'lmbda': self.lmbda,
            'alpha': self.act_fn.alpha,
            'performance': self.performance,
            'thetas': [t.tolist() for t in self.thetas],
            'biases': [b.tolist() for b in self.biases]
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        wandb.save(filename)

    @staticmethod
    def load(filename):
        with open(filename) as f:
            data = json.load(f)

        n = NeuralNetwork(
            data['ns'], data['eta'], data['lmbda'], data['alpha']
        )
        n.thetas = [np.array(t) for t in data['thetas']]
        n.biases = [np.array(b) for b in data['biases']]
        n.performance = data['performance']
        return n

    def compute_metrics(self, data):
        # Return the total number of correct predictions and total loss
        # on the provided dataset. Used for learning curves.
        correct = loss = 0
        for x, y in data:
            output = self.predict(x)
            correct += np.argmax(y) == np.argmax(output)
            loss += log_loss(y, output)
        loss = 100 * loss / len(data)
        return correct, loss

    def report_metrics(self, data, label, i):
        correct, loss = self.compute_metrics(data)
        size = len(data)
        percentage = 100 * correct / size

        # Log the data in wandb and also locally.
        print(f'Epoch {i} {label}  acc: {correct} / {size} ({percentage}%)')
        print(f'Epoch {i} {label} loss: {loss}')
        wandb.log({'epoch': 0, f'{label}_accuracy': percentage})
        wandb.log({'epoch': 0, f'{label}_loss': loss})

        return percentage, loss

    def get_activations(self, x):
        # Validate the size of the input.
        self.validate_input(x)

        # We scale down the input to be in [0, 1].
        # Also, we add a 1 to the end to account for the bias.
        activations = [x]

        # Iterate over each layer, excluding the output layer
        for theta, bias in zip(self.thetas[:-1], self.biases[:-1]):
            # The input to this layer is the matrix product of the weights
            # associated with this layer and the activations of the previous
            # layer plus the biases.
            z = np.array(
                np.dot(theta, activations[-1]) + bias, dtype='float64'
            )

            # Apply the activation (LReLU) function to get the activations
            # for this layer, and add a 1 to the end for the bias.
            activations.append(self.act_fn.f(z))

        # For the output layer, we apply the softmax function, computed as:
        # exp(z) / sum(exp(z in zs))
        # The sum of outputs is clearly 1, which gives us
        # a useful interpretation as the 'confidence' for each possible output.
        z = np.dot(self.thetas[-1], activations[-1]) + self.biases[-1]
        activations.append(softmax(z))
        return activations

    def train_batch(self, batch):
        delta_thetas = [
            np.zeros((self.ns[i], self.ns[i - 1]))
            for i in range(1, self.n)
        ]
        delta_biases = [np.zeros((x,)) for x in self.ns[1:]]

        # Iterate over all examples in the batch
        for x, y in batch:
            # Activations for the current training example
            activations = self.get_activations(x)

            # We can trivially compute the bias and weight derivatives
            # for the last layer with the help of y and the prediction.
            # These formulae are derived by applying the chain rule to
            # the softmax (activation) and the log loss (error) functions.
            difference = np.array(activations[-1] - y)
            delta_biases[-1] += difference
            delta_thetas[-1] += np.dot(
                difference.reshape((-1, 1)),
                activations[-2].reshape(1, -1)
            )

            # The derivatives array stores the derivatives of the error
            # function
            # with respect to the activations.
            # This array is used for backpropagation.
            derivatives = [np.zeros((x,)) for x in self.ns]
            # Derivative of the second last layer
            # i.e, layer before output layer
            # Change to [-2] or in init make self.ns[1:-1] since
            # input and output layers won't have derivatives.
            derivatives[-2] = np.dot(self.thetas[-1].transpose(), difference)

            # Loop over all hidden layers, backwards
            for i in range(self.n - 3, -1, -1):
                # Assign some variables to make the following code cleaner.
                a_t = activations[i].reshape((1, -1))
                theta_t = self.thetas[i].transpose()

                # We compute a term called the "error".
                # This can be considered similar to the "difference" term above
                # with the derivative of the activation function multiplied.
                error = derivatives[i + 1] * self.act_fn.d(activations[i + 1])
                error_v = error.reshape((-1, 1))

                # The following formulae used to compute the derivatives are
                # derived using the chain rule.
                # Our goal is to compute the derivatives of the error function
                # with respect to each parameter, to tweak them as necessary.
                # These values are added so as to 'accumulate' them
                # over the batch we are currently training on.
                delta_biases[i] += error
                delta_thetas[i] += np.dot(error_v, a_t)
                derivatives[i] = np.dot(theta_t, error)

        scale_factor = self.eta / len(batch)
        for i in range(self.n - 1):
            # L2 regularization term
            self.thetas[i] *= 1 - (scale_factor * self.lmbda)

            # Updates
            self.thetas[i] -= scale_factor * delta_thetas[i]
            self.biases[i] -= scale_factor * delta_biases[i]

    def validate_input(self, x):
        # This is hardly needed - it was added to serve as a sanity check
        # in the event someone forgot to "flatten" the inputs.
        if len(x) != self.ns[0]:
            raise ValueError(
                'Number of inputs != number of input neurons! '
                f'Expected: {self.ns[0]}, received: {len(x)}'
            )


def main():
    # TODO: Animated plotting to show performance change with time.
    #  check scratch.py
    wandb.init(project='digits')
    n = NeuralNetwork([784, 128, 128, 10])
    training, validation, test = load_data()
    tick = time.perf_counter()
    n.train(training[:20], validation[:10], 1)
    tock = time.perf_counter()
    print(tock - tick)
    # n.plot()

    # for i in range(5):
    #     draw_digit(training[i][0])


if __name__ == '__main__':
    main()
