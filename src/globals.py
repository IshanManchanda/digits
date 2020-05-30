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
