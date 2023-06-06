# -*- coding: utf-8 -*-
"""
This module contains the definition of the Platform object, which is the core
of the microscope package. It coordinates hardware to perform high-level
operations between the microscope, the DMD, and the Arduinos.

"""
import time

import cv2
import numpy as np
from pycromanager import Core, Studio

DEVICE_CAMERA = 'Camera-1'
DEVICE_TIGER = 'TigerCommHub'
DEVICE_XY_STAGE = 'XYStage:XY:31'
DEVICE_Z_STAGE = 'ZStage:Z:32'
DEVICE_LED1 = 'LED:L:37'  # unclear

PROPERTY_LED_STATE = 'State'
VALUE_LED_OPEN = 'Open'
VALUE_LED_CLOSE = 'Closed'


class MMCPlatform:
    '''
    Main class for the microscope, DMD, neopixel, microfluidics, etc...
    Also MM GUI. Everything hardware basically (except GPU, computer etc)
    '''

    def __init__(self):

        print("#### Initializing platform")

        # Core properties:
        print("\n## Connecting to micromanager")
        # self.bridge = Bridge()
        "Python-Java bridge to communicate with micromanager"
        self.mmc = Core()
        "Object mirroring micromanager-core (core hardware functions)"
        self.studio = Studio()
        "Object mirroring micromanager-studio (GUI)"
        print(self.mmc.get_version_info())
        print(self.mmc.get_api_version_info())
        print(self.studio.get_sys_config_file())

    # %% Actions
    def snap(self):
        """
        Snap image (ie trigger lamp if auto-shutter is on, trigger camera
        exposure, wait for camera end of acquistion signal, turn off lamp)

        Returns
        -------
        None.

        """

        self.mmc.snap_image()

    def retrieve(self):
        """
        Retrieve image from camera.

        Returns
        -------
        pixels : 2D numpy array of uint16
            Snapped image. Dimensions are (2048, 2048) unless camera ROI or
            binning properties have been changed. Note that the datatype of the
            array is always going to be uint16, even if the camera bitdepth was
            set to 12 bit.

        """

        tagged_image = self.mmc.get_tagged_image()
        pixels = np.reshape(
            tagged_image.pix,
            newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']]
        )
        return pixels

    def focus(self, mode='fullfocus'):
        """
        Focus the PFS

        Parameters
        ----------
        mode : str, optional
            Focussing mode, 'fullfocus', 'continuous', or 'off'. If 'fullfocus'
            and continuous mode has been turned off, the PFS will be turned on,
            focussed, and then turned back off. If 'fullfocus' and continuous
            mode has been turn on, behavior is similar to 'continuous'. If
            'continuous' this function turns continuous focus on if it is off
            and then checks that the focus is locked. If 'off' continuous focus
            is turned off
            The default is 'fullfocus'.

        Note : Setting Z (e.g. by using Platform.move() with a z value) will
        turn off continuous focussing. This is a hardware limitation of the
        Ti2 there is nothing we can do about it.

        Returns
        -------
        None.

        """

        mode = mode.lower()

        # Turn on continuous mode if not enabled:
        if mode == 'continuous' and not self.mmc.is_continuous_focus_enabled():
            self.mmc.enable_continuous_focus(True)

        if mode in ('fullfocus', 'continuous'):
            try:
                # This function turns PFS on, focusses, and then back off in
                # non-continuous mode, and just waits until the autofocus
                # converges in continuous mode:
                self.mmc.full_focus()
            except Exception as em:
                # If autofocus errors out, which happens from time to time,
                # just print error and keep going
                print(em)
            return

        # Turn off continuous focus:
        if mode == 'off':
            self.mmc.enable_continuous_focus(False)

    def channel(self, channel_name='Trans'):
        """
        Set imaging channel configuration
        (for more details, see configurations in MM)

        Parameters
        ----------
        channel_name : str, optional
            The name of the channel to set up. The default is 'Trans'.

        Returns
        -------
        None.

        """
        self.mmc.set_config('Channel', channel_name)

    def get_channel(self):
        """
        Get the name of the current channel config

        Returns
        -------
        str
            Channel name.

        """
        return self.mmc.get_current_config('Channel')

    def exposure(self, exposure_time):
        """
        Set camera exposure time.

        Parameters
        ----------
        exposure_time : int
            Exposure, in ms.

        Returns
        -------
        None.

        """
        self.mmc.set_exposure(exposure_time)

    def get_exposure(self):
        """
        Get current camera exposure.

        Returns
        -------
        int
            Exposure, in ms.

        """
        return self.mmc.get_exposure()

    def hw_acquire(self, channels, exposures):
        """
        Trigger arduino to set & acquire a certain acquisition sequence
        For now this only works for trans and fluo channels

        Parameters
        ----------
        channels : List of str
            Names of channels to trigger. Note that it does not actually set
            the channels in micromanager, ie the filter cubes are not set
            in position. These are only used to determine which lamp to trigger
            via the arduino.
        exposures : List of int
            Camera exposure values to use for each channel. Ideally you would
            want to use the same value for each channel if using multiple, as
            the exposure does not actually get changed in the camera's settings
            if it is the same as before. Otherwise it adds some overhead that
            might nullify the speed increase offered by hardware triggering.

        Returns
        -------
        images: List of 2D numpy arrays of uint16
            The images for each channel

        """

        for channel, exposure in zip(channels, exposures):
            # Set exposure:
            self.exposure(exposure)

            # Trigger snap:
            # self.synchronizer.snap(channel=channel)

            # Wait for image:
            # while not self.synchronizer.camera_ready():
            #    time.sleep(.0005)

        # Retrieve images: (much faster to acquire first and then retrieve)
        images = []
        for channel, exposure in zip(channels, exposures):
            images.append(self.retrieve())

        return images

    # %% Device properties
    def get(self, device, property):
        """
        Get the value of a certain device property.
        See micromanager's devices and properties browser for more information.

        Parameters
        ----------
        device : str
            Name of device of interest.
        property : str
            Name of device's property.

        Returns
        -------
        str, int, or float
            Device property value.

        """
        return self.mmc.get_property(device, property)

    def set(self, device, property, value):
        """
        Set the value of a certain device property.
        See micromanager's devices and properties browser for more information.

        Parameters
        ----------
        device : str
            Name of device of interest.
        property : str
            Name of device's property.
        value : str, int, or float
            Device property value.

        Returns
        -------
        None.

        """
        self.mmc.set_property(device, property, value)

    def get_all_properties(self):
        """
        Get all properties, for all devices.

        Returns
        -------
        all_properties : dict
            All properties of all devices, as nested dicts.

        """

        # Init:
        all_properties = dict()

        # Get all MM devices:
        devices = self._MMdevices()

        # Get MM device properties:
        for device in devices:
            all_properties[device] = self._device_properties(device)

        return all_properties

    def get_available_channels(self):
        """
        Get the list of available channel configurations

        Returns
        -------
        channels : list of str
            Available channels.

        """

        channels_vec = self.mmc.get_available_configs('Channel')

        channels = []
        for channel_nb in range(channels_vec.size()):
            channels += [channels_vec.get(channel_nb)]

        return channels

    def _MMdevices(self):
        """
        Get list of all MM devices

        Returns
        -------
        devices : TYPE
            DESCRIPTION.

        """

        # Init:
        devices = []

        # Get Micro manager's devices list:
        dev_strvec = self.mmc.get_loaded_devices()

        # Turn into a list of str:
        for dev_ind in range(dev_strvec.size()):
            devices.append(dev_strvec.get(dev_ind))

        return devices

    def _device_properties(self, device):
        """
        Get all properties of a single device

        Parameters
        ----------
        device : str
            Device name.

        Returns
        -------
        properties : dict
            names and values of device properties.

        """

        # Init:
        properties = dict()

        # Get all device property names:
        property_names = self.mmc.get_device_property_names(device)

        # Retrieve property values:
        for prop_number in range(property_names.size()):
            prop_name = property_names.get(prop_number)
            properties[prop_name] = self.mmc.get_property(device, prop_name)
            time.sleep(.01)

        return properties

    # %% MM Studio interface:

    def live(self, mode=True):
        """
        Turn live camera acquisition on/off

        Parameters
        ----------
        mode : bool, optional
            Whether to turn live mode on or off. The default is True.

        Returns
        -------
        None.

        """
        self.studio.live().set_live_mode(mode)

    def positions_list(self):
        """
        Retrieve positions list from MM interface

        Returns
        -------
        positions_list : list of dict
            List of positions as they would be returned by the where() method.

        """

        # Get list from MM studio:
        studio_list = self.studio.get_positions_list()
        positions_list = []

        # Run through positions an retrieve xyz:
        for pos_nb in range(studio_list.get_number_of_positions()):
            studio_position = studio_list.get_position(pos_nb)
            position = dict()
            position['x'] = studio_position.get_x()
            position['y'] = studio_position.get_y()
            position['z'] = studio_position.get_z()
            # TODO
            position['pfs'] = None
            positions_list.append(position)

        return positions_list

    def get_xy_stage_limits_um(self):
        xmin_key = 'LowerLimX(mm)'
        xmax_key = 'UpperLimX(mm)'
        ymin_key = 'LowerLimY(mm)'
        ymax_key = 'UpperLimY(mm)'
        properties = self._device_properties(DEVICE_XY_STAGE)
        return ((float(properties[xmin_key]) * 1000, float(properties[xmax_key]) * 1000),
                (float(properties[ymin_key]) * 1000, float(properties[ymax_key]) * 1000))


