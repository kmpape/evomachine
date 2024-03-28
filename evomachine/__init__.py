import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

delta_path = os.path.join(current_dir, '..', '..', 'de-lta-rt')
sys.path.append(delta_path)

asitiger_path = os.path.join(current_dir, '..', '..', 'asitiger')
sys.path.append(asitiger_path)

sync_board_path = os.path.join(current_dir, '..', '..', 'sync_board')
sys.path.append(sync_board_path)
