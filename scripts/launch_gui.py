import glob
import os
import sys
from PyQt5.QtWidgets import QApplication

sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/asitiger')
sys.path.append(os.path.expanduser('~') + '/workspace_python/conda_evomachine3.9/evomachine_repo')

from evomachine.acquisition import EvoCamera, TestCamera
from evomachine.config import ConfigCRISP, ConfigFocus, ConfigFocusAlgorithm, \
    CRISP_CONFIG_DEFAULT, FOCUS_CONFIG_DEFAULT, IMAGE_CONFIG_DEFAULT, DEVICE_CONFIG_EVO_TEST,\
    OBJECTIVE_CONFIG_AIR,  OBJECTIVE_CONFIG_OIL, CRISP_CONFIG_OIL
from evomachine.dmd import DMDControl
from evomachine.gui import EvoGUI


if __name__ == '__main__':
    is_testmode = False
    is_oil_objective = False

    if is_testmode:
        filenames = sorted(glob.glob("/mnt/ImageData/Scott/2023-12-08/*.tiff"))
        cam = TestCamera(
            cfg_device=DEVICE_CONFIG_EVO_TEST,
            cfg_objective=OBJECTIVE_CONFIG_OIL if is_oil_objective else OBJECTIVE_CONFIG_AIR,
            cfg_image=IMAGE_CONFIG_DEFAULT,
            cfg_crisp=CRISP_CONFIG_DEFAULT,
            cfg_focus=FOCUS_CONFIG_DEFAULT,
            filenames=filenames,
            pos_to_filename=None,
        )
    else:
        cam = EvoCamera(
            cfg_device=DEVICE_CONFIG_EVO_TEST,
            cfg_image=IMAGE_CONFIG_DEFAULT,
            cfg_objective=OBJECTIVE_CONFIG_OIL if is_oil_objective else OBJECTIVE_CONFIG_AIR,
            cfg_focus=FOCUS_CONFIG_DEFAULT,
            cfg_crisp=CRISP_CONFIG_DEFAULT,
        )
    dmd = DMDControl()

    app = QApplication(sys.argv)
    w = EvoGUI(cam, dmd)
    w.show()
    sys.exit(app.exec_())
