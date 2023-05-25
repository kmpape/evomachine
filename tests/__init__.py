import sys
import os

# Get the parent directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the path of the 'delta' project to sys.path
delta_path = os.path.join(current_dir, '..', '..', 'de-lta-rt')
sys.path.append(delta_path)