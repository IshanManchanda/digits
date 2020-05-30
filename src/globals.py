import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
networks_dir = os.path.join(root_dir, 'networks')
current_dir = os.path.join(networks_dir, 'current')
archive_dir = os.path.join(networks_dir, 'archive')
