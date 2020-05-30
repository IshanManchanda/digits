import os

# Project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

networks_dir = os.path.join(root_dir, 'networks')
# Directory for the current run
current_dir = os.path.join(networks_dir, 'current')
# Directory for previous runs
archive_dir = os.path.join(networks_dir, 'archive')
