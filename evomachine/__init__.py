import sys
import os
import logging
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))

delta_path = os.path.join(current_dir, '..', '..', 'de-lta-rt')
sys.path.append(delta_path)

asitiger_path = os.path.join(current_dir, '..', '..', 'asitiger')
sys.path.append(asitiger_path)

sync_board_path = os.path.join(current_dir, '..', '..', 'sync_board')
sys.path.append(sync_board_path)

# Set TF logging level:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import tensorrt here otherwise TF does not find it
import tensorrt

# Set matplotlib loglevel
plt.set_loglevel("warning")

logging.basicConfig(level=logging.INFO)

# Disable numba debugging messages
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
