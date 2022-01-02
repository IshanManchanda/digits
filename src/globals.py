import os

# Project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

networks_dir = os.path.join(root_dir, 'networks')
if not os.path.isdir(networks_dir):
    os.mkdir(networks_dir)

# Directory for the current run
current_dir = os.path.join(networks_dir, 'current')
if not os.path.isdir(current_dir):
    os.mkdir(current_dir)

# Directory for previous runs
archive_dir = os.path.join(networks_dir, 'archive')
if not os.path.isdir(archive_dir):
    os.mkdir(archive_dir)

data_dir = os.path.join(root_dir, 'data')
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# The _py3 version of the dataset is a redumped version for Python 3
# which doesn't use Python 2's latin1 encoding
mini_path = os.path.join(data_dir, 'mnist_py3_mini_deskewed.pkl.gz')
deskew_path = os.path.join(data_dir, 'mnist_py3_deskewed.pkl.gz')
mnist_path = os.path.join(data_dir, 'mnist_py3.pkl.gz')

network_path = os.path.join(current_dir, 'network.json')
