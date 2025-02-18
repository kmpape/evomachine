from dataclasses import dataclass, field

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'asitiger'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'de-lta-rt'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sync_board'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from evomachine.evotypes import LEDType
from evomachine.config import ConfigCamera, ConfigCRISP, ObjectiveConfigType, ImageConfigType, ConfigFocus, FilterWheelType, FocusAlgorithmType

from pathlib import Path

import pickle

from enum import Enum, auto
class MessageType(Enum):
    text = auto()
    image = auto()
    camera_config = auto()
    crisp_status = auto()
    crisp_initialised = auto()
    set_led = auto()
    request_camera_config = auto()
    check_crisp_status = auto()
    set_crisp = auto()
    init_crisp = auto()
    request_frame = auto()

@dataclass
class Message:
    content: str
    type: MessageType

    def encode(self) -> bytes:
        """Encode the message to a bytes string."""
        return pickle.dumps(self)

    @staticmethod
    def decode(data: bytes) -> 'Message':
        """Decode bytes to a Message object."""
        return pickle.loads(data)

# Test
text_message = Message(content="Hello, world!", type=MessageType.text)
encoded_message = text_message.encode()
decoded_message = Message.decode(encoded_message)
print(decoded_message)