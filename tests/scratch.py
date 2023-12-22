import sys


sys.path.append('/home/hslab/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append('/home/hslab/workspace_python/conda_evomachine3.9/evomachine_repo')


from evomachine.acquisition import EvoCamera
from evomachine.config import DEVICE_CONFIG_EVO_TEST

bla = EvoCamera(DEVICE_CONFIG_EVO_TEST)
bla._move_stage(0)
