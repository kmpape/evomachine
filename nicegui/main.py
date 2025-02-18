from nicegui import ui, events, app, binding, run

import asyncio
from websockets.asyncio.client import connect
from websockets import ClientConnection

from messages import *
from evomachine.evotypes import LEDType

import numpy as np
from PIL import Image

RELOAD = True

websocket_connection: ClientConnection = None
websocket_state = {
    'connected': False
}

# Better performance than using dictionary
@binding.bindable_dataclass
class DisplayImage:
    source: Image = field(default_factory=lambda: Image.fromarray((np.random.rand(360, 640) * 255).astype('uint8')))

display_img = DisplayImage()

# Crisp status enum
from enum import Enum, auto
class CRISPInitStatus(Enum):
    NOTINIT = auto()
    INITING = auto()
    INITED  = auto()

led_brightness = {}
crisp_status   = {'locked': None, 'init': CRISPInitStatus.NOTINIT}

camera_config = None
camera_config_event = asyncio.Event()

# def convert_frame(frame):
#     return Image.fromarray(((frame/1024) * 255).astype('uint8'))

async def on_message(message):
    global camera_config, display_img
    message = await run.cpu_bound(Message.decode, message)
    print(f"Received message: {message.type}")

    if message.type == MessageType.camera_config:
        camera_config = message.content
        camera_config_event.set()
    elif message.type == MessageType.crisp_status:
        crisp_status['locked'] = message.content
    elif message.type == MessageType.text:
        print(f"GUI: Received text message: {message.content}")
    elif message.type == MessageType.crisp_initialised:
        x = message.content
        crisp_status['init'] = CRISPInitStatus.INITED if x else CRISPInitStatus.NOTINIT
    # elif message.type == MessageType.image:
        # display_img.source = await run.cpu_bound(convert_frame, message.content)
    else:
        print(f"GUI: Unhandled message type: {message}")


async def connect_to_websocket():
    global websocket_connection, websocket_state
    while websocket_connection is None:
        try:
            websocket_connection = await connect("ws://localhost:8765", max_size=None)
            websocket_state['connected'] = True
            print("Connected to websocket")

            async def receive_messages():
                async for message in websocket_connection:
                    await on_message(message)

            asyncio.create_task(receive_messages())
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 1 second.")
            await asyncio.sleep(1)

    await websocket_connection.send(Message(type=MessageType.request_camera_config, content="").encode())

# async def hello():
#     global websocket_connection
#     if websocket_connection is None:
#         print("Not connected to websocket!")
#         return
#     await websocket_connection.send(TextMessage("Hello world!").encode())

async def set_led(led: LEDType, brightness: float):
    global websocket_connection, led_brightness
    if websocket_connection is None:
        print("Not connected to websocket!")
        return
    await websocket_connection.send(Message(type=MessageType.set_led, content=(led, brightness)).encode())

async def request_crisp_status():
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket!")
        return
    await websocket_connection.send(Message(type=MessageType.check_crisp_status, content="").encode())

async def request_lock_crisp(lock):
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket")
        return
    await websocket_connection.send(Message(type=MessageType.set_crisp, content=lock).encode())

async def request_init_crisp():
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket")
        return
    crisp_status['init'] = CRISPInitStatus.INITING
    await websocket_connection.send(Message(type=MessageType.init_crisp, content="").encode())

async def request_frame():
    global websocket_connection
    print("Requesting frame")
    if websocket_connection is None:
        print("Not connected to websocket")
        return
    await websocket_connection.send(Message(type=MessageType.request_frame, content="").encode())    

@ui.page('/')
async def page():
    await camera_config_event.wait() # Make sure the camera config is received before rendering the page

    with ui.row():
        with ui.card().tight():
            with ui.card_section().classes('w-full bg-gray-200'):
                ui.label("LED Controls")
            with ui.card_section():
                for led in camera_config.leds:
                    with ui.row().classes('w-full no-wrap'):
                        led_brightness.setdefault(led.name, 29.0)
                        ui.label(f"{led.name.replace('_', ' ')}").classes('w-1/4')
                        ui.number(value=led_brightness[led.name], min=0, max=100, step=1)\
                            .bind_value(led_brightness, led.name)\
                            .on(type='keydown.enter', handler=lambda led=led: set_led(led, led_brightness[led.name]))\
                            .classes('w-1/4')
                        ui.button("Set LED", on_click=lambda led=led: set_led(led, led_brightness[led.name]))\
                            .classes('w-2/4')

        with ui.card().tight():
            with ui.card_section().classes('w-full bg-gray-200'):
                ui.label("CRISP Controls")
            with ui.card_section():
                ui.label("CRISP status: ")
                ui.label(text="").bind_text_from(crisp_status, 'locked', backward=lambda x: "Locked" if x==True else ("Unlocked" if x==False else "Unknown"))
                ui.label(text="").bind_text_from(crisp_status, 'init', backward=lambda x: "Initialised" if x==CRISPInitStatus.INITED else ("Initialising" if x==CRISPInitStatus.INITING else "Not initialised"))
                with ui.row():
                    ui.button("Check CRISP", on_click=request_crisp_status)
                    ui.button("Lock CRISP", on_click=lambda: request_lock_crisp(True))
                    ui.button("Init CRSIP", on_click=request_init_crisp)
                    ui.button("Unlock CRISP", on_click=lambda: request_lock_crisp(False))
    
    with ui.card().tight():
        with ui.card_section().classes('w-full bg-gray-200'):
            ui.label("Image Display")
        with ui.card_section():
            with ui.column():
                ui.button("Get Frame", on_click=request_frame)
                # ui.interactive_image(source='http://127.0.0.1:8081/')
                # ui.html("<canvas id='imageCanvas'></canvas>")
                # img = ui.interactive_image()
                ui.html("<img id='image' width='800'>")
                ui.run_javascript("""
                    const ws = new WebSocket("ws://localhost:8000/ws");
                    ws.binaryType = 'arraybuffer'
                    ws.onmessage = async event => {
                    const message = new Uint8Array(event.data);
                    // Handle binary image data
                    const blob = new Blob([message], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(blob);
                    const img = document.getElementById('image');
                    img.src = url;
                };""")
                
async def disconnect_websocket():
    global websocket_connection
    if websocket_connection is None:
        return
    if websocket_state['connected'] == False:
        return
    await websocket_connection.close()

def main():
    global websocket_connection

    # If reload is enabled, then it has to run the main program as a subprocess (https://github.com/zauberzeug/nicegui/issues/794)
    # => it runs the file again as a subprocess causing everything to be executed twice
    # => This reload runner doesn't need to run the actual code
    if RELOAD and __name__ == '__main__':
        print("Starting UI in subprocess")
        ui.run(host='127.0.0.1', port=8080, reload=RELOAD, reconnect_timeout=10)
        return

    app.on_startup(connect_to_websocket)
    app.on_shutdown(disconnect_websocket)

    ui.run(host='127.0.0.1', port=8080, reload=RELOAD, reconnect_timeout=10)

if __name__ in ['__main__', '__mp_main__']:
    main()