import glob
import os
import queue
import sys
from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/de-lta-rta')

from delta.config import DEFAULT_CONFIG_MOTHERMACHINE

from evomachine.acquisition import EvoCamera, TestCamera
from evomachine.automaton import Automaton
from evomachine.config import ConfigCRISP, ConfigFocus, ConfigFocusAlgorithm, \
    CRISP_CONFIG_DEFAULT, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, DEVICE_CONFIG_EVO_TEST,\
    OBJECTIVE_CONFIG_AIR,  OBJECTIVE_CONFIG_OIL, CRISP_CONFIG_OIL
from evomachine.dmd import DMDControl
from evomachine.gui import EvoGUI
from evomachine.strategy import BasicStrategy   # TODO add dropdown in GUI


if __name__ == '__main__':
    is_oil_objective = False
    cam = EvoCamera(
        cfg_device=DEVICE_CONFIG_EVO_TEST,
        cfg_image=IMAGE_CONFIG_DEFAULT,
        cfg_objective=OBJECTIVE_CONFIG_OIL if is_oil_objective else OBJECTIVE_CONFIG_AIR,
        cfg_focus=FOCUS_CONFIG_DEFAULT,
        cfg_crisp=CRISP_CONFIG_DEFAULT,
    )
    dmd = DMDControl()
    data_queue = queue.Queue()
    strategy = BasicStrategy()  # TODO add dropdown in GUI
    automaton = Automaton(
        cfg_device=DEVICE_CONFIG_EVO_TEST,
        cfg_image=IMAGE_CONFIG_DEFAULT,
        cfg_delta=DEFAULT_CONFIG_MOTHERMACHINE,
        cfg_focus=FOCUS_CONFIG_DEFAULT,
        camera=cam,
        dmd=dmd,
        strategy=strategy,
        data_queue=data_queue,
        use_segmentation=False,
    )

    app = QApplication(sys.argv)
    w = EvoGUI(cam=cam, dmd=dmd, data_queue=data_queue, automaton=automaton)
    w.show()
    sys.exit(app.exec_())
