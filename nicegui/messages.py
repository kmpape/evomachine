from dataclasses import dataclass, field
import json
import numpy as np

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'asitiger'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sync_board'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from evomachine.evotypes import LEDType
from evomachine.config import ConfigCamera, ConfigCRISP, ObjectiveConfigType, ImageConfigType, ConfigFocus, FilterWheelType, FocusAlgorithmType

from pathlib import Path

@dataclass
class Message:
    content: str
    type: str

    def encode(self) -> bytes:
        """Encode the message to a JSON string and then to bytes."""
        return json.dumps(self.__dict__).encode('utf-8')

    @staticmethod
    def decode(data: bytes) -> 'Message':
        """Decode bytes to a JSON string and then to a Message object."""
        json_data = json.loads(data.decode('utf-8'))
        return Message(**json_data)

# Example usage:
@dataclass
class TextMessage(Message):
    content: str = ""
    type: str = "text"

@dataclass
class ImageMessage(Message):
    content: str = ""
    url: str = ""
    type: str = "image"

@dataclass
class LEDStatusMessage(Message):
    content: dict = field(default_factory=dict)
    type: str = "ledstatus"

@dataclass
class SetLEDMessage(Message):
    content: tuple[LEDType, float] = field(default_factory=tuple)
    type: str = "setled"

    def encode(self) -> bytes:
        """Have to handle this manually to be able to use LEDType"""
        return json.dumps({
            'content': (self.content[0].name, self.content[1]),
            'type': self.type
        }).encode('utf-8')

    @staticmethod
    def decode(data: bytes) -> 'SetLEDMessage':
        """Have to handle this manually to be able to use LEDType"""
        json_data = json.loads(data.decode('utf-8'))
        led = LEDType[json_data['content'][0]]
        brightness = json_data['content'][1]
        return SetLEDMessage(content=(led, brightness))

@dataclass
class RequestConfigCameraMessage(Message):
    content: str = ""
    type: str = "requestcameraconfig"

@dataclass
class CameraConfigMessage(Message):
    content: ConfigCamera = field(default_factory=ConfigCamera)
    type: str = "cameraconfig"

    def encode(self) -> bytes:
        content = {}
        content['objective'] = self.content.objective.__dict__.copy() # ObjectiveConfigType
        content['image']     = self.content.image.__dict__.copy()     # ImageConfigType
        content['image']['pxl_dtype'] = str(self.content.image.pxl_dtype)
        
        content['focus']     = self.content.focus.__dict__.copy() # ConfigFocus
        content['focus']['focus_channel'] = self.content.focus.focus_channel.name
        content['focus']['algorithm'] = self.content.focus.algorithm.name
        
        content['autofocus'] = self.content.autofocus.__dict__.copy() # ConfigCRISP

        content['leds']      = [x.name for x in self.content.leds] # List[LEDType]
        content['filters']   = [x.name for x in self.content.filters] # List[FilterWheelType]
        content['path_to_save'] = str(self.content.path_to_save)
        content['default_exposure_time'] = self.content.default_exposure_time
        content['default_focus_channel_id'] = self.content.default_focus_channel_id
        content['cam_pxl_size'] = self.content.cam_pxl_size
        return json.dumps({
            'content': content,
            'type': self.type
        }).encode('utf-8')

    @staticmethod
    def decode_content(json_data: dict) -> 'CameraConfigMessage':
        json_data['image']['pxl_dtype'] = np.dtype(json_data['image']['pxl_dtype'])
        image = ImageConfigType(**json_data['image'])
        json_data['focus']['focus_channel'] = LEDType[json_data['focus']['focus_channel']]
        
        json_data['focus']['algorithm'] = FocusAlgorithmType[json_data['focus']['algorithm']]
        focus = ConfigFocus(**json_data['focus'])

        autofocus = ConfigCRISP(**json_data['autofocus'])
        objective = ObjectiveConfigType(**json_data['objective'])
        leds = [LEDType[x] for x in json_data['leds']]
        filters = [FilterWheelType[x] for x in json_data['filters']]
        path_to_save = Path(json_data['path_to_save'])
        default_exposure_time = float(json_data['default_exposure_time'])
        default_focus_channel_id = int(json_data['default_focus_channel_id'])
        cam_pxl_size = float(json_data['cam_pxl_size'])
        return CameraConfigMessage(content=ConfigCamera(
            autofocus=autofocus,
            image=image,
            focus=focus,
            objective=objective,
            leds=leds,
            filters=filters,
            path_to_save=path_to_save,
            default_exposure_time=default_exposure_time,
            default_focus_channel_id=default_focus_channel_id,
            cam_pxl_size=cam_pxl_size,
        ))

# led_message = LEDStatusMessage(content={'LED1':True, 'LED2':False})
# encoded_message = led_message.encode()
# decoded_message = led_message.decode(encoded_message)
# print(decoded_message)

# Encoding and decoding example
# text_message = TextMessage(content="Hello, world!")
# encoded_message = text_message.encode()
# decoded_message = Message.decode(encoded_message)

# print(decoded_message)