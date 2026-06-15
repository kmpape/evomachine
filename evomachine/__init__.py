import os
import logging
import matplotlib.pyplot as plt

# Set TF logging level:
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Set matplotlib loglevel
plt.set_loglevel("warning")

logging.basicConfig(level=logging.INFO)

# Disable numba debugging messages
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
