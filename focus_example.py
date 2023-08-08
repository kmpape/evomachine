from time import sleep
from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion.units import Units
from zaber_motion.device_db_source_type import DeviceDbSourceType
import cv2 as cv
from simple_pyspin import Camera

# The device that moves the objective up and down
Z_AXIS_DEVICE = 3
# The absolute position to start the focus search at
START_MM = 19.83
# The absolute position to end the search at
END_MM = 19.855
# The granularity of the search
STEP_SIZE_MM = 0.001
# Set to True if the program should print focus score and image at each step
SHOW_STEP_INFO = False

Library.set_device_db_source(DeviceDbSourceType.WEB_SERVICE, 'https://api.zaber.io/device-db/master')

with Connection.open_serial_port("/dev/cu.usbserial-AC01ZOZU") as connection:
  # initialize the microscope actuators
  device_list = connection.detect_devices()
  # TODO: This might be a controller instead of a device, in which case should have axis number too
  z_axis = connection.get_device(Z_AXIS_DEVICE).get_axis(1)
  if not z_axis.is_homed():
    z_axis.home()
  z_axis.move_absolute(START_MM, Units.LENGTH_MILLIMETRES)

  with Camera() as cam:
    # Set the camera to take individual shots
    cam.AcquisitionMode = 'SingleFrame'
    # To control the exposure settings, we need to turn off auto
    cam.GainAuto = 'Off'
    # Set the gain to maximum to achieve best contrast
    cam.Gain = cam.get_info('Gain')['max']
    cam.ExposureAuto = 'Off'
    cam.ExposureTime = 20000 # microseconds

    best_focus_score = 0
    best_focus_position = 0
    while True:
      z_axis.move_relative(STEP_SIZE_MM, Units.LENGTH_MILLIMETRES)
      position = z_axis.get_position(Units.LENGTH_MILLIMETRES)
      cam.start()
      image_raw = cam.get_array()
      cam.stop()
      image = cv.medianBlur(image_raw,3)
      laplacian = cv.Laplacian(image_raw, cv.CV_64F)
      focus_score = laplacian.var()
      if focus_score > best_focus_score:
        best_focus_position = position
        best_focus_score = focus_score
      if SHOW_STEP_INFO:
        print(f'focus {position}: {focus_score}')
        cv.imshow(f'Raw Image {position} ({focus_score})', image_raw)
      if position > END_MM:
        break
    
    z_axis.move_absolute(best_focus_position, Units.LENGTH_MILLIMETRES)
    cam.start()
    image_raw = cam.get_array()
    cam.stop()
    cv.imshow(f'Best Image', image_raw)
    print(f'The best focus ({best_focus_score}) was found at {best_focus_position}. Press any key to exit.')
    cv.waitKey(0)
    cv.destroyAllWindows()
