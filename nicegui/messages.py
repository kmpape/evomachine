from dataclasses import dataclass, field
import json

@dataclass
class Message:
    type: str
    content: str

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
    type: str = "text"
    content: str = ""

@dataclass
class ImageMessage(Message):
    type: str = "image"
    content: str = ""
    url: str = ""

@dataclass
class LEDStatusMessage(Message):
    type: str = "ledstatus"
    content: dict = field(default_factory=dict)

led_message = LEDStatusMessage(content={'LED1':True, 'LED2':False})
encoded_message = led_message.encode()
decoded_message = led_message.decode(encoded_message)
print(decoded_message)

# # Encoding and decoding example
# text_message = TextMessage(content="Hello, world!")
# encoded_message = text_message.encode()
# decoded_message = Message.decode(encoded_message)

# print(decoded_message)