import sys
import os
import logging
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

delta_path = str(WORKSPACE_ROOT / "de-lta-rt")
sys.path.append(delta_path)

asitiger_path = str(WORKSPACE_ROOT / "asitiger")
sys.path.append(asitiger_path)

sync_board_path = str(WORKSPACE_ROOT / "sync_board")
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
