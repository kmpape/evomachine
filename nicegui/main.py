import nicegui.ui as ui
import nicegui.app as app

import asyncio
from websockets.asyncio.client import connect
from websockets import ClientConnection

from messages import *
from evomachine.evotypes import LEDType

RELOAD = True

websocket_connection: ClientConnection = None
websocket_state = {
    'connected': False
}

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

async def on_message(message):
    global camera_config
    message = Message.decode(message)
    # print(f"Received message: {message}")

    if message.type == CameraConfigMessage.type:
        camera_config = CameraConfigMessage.decode_content(message.content)
        camera_config_event.set()
    elif message.type == CRISPStatusMessage.type:
        crisp_status['locked'] = CRISPStatusMessage.decode_content(message.content)
    elif message.type == TextMessage.type:
        print(f"GUI: Received text message: {message.content}")
    elif message.type == CRISPInitialisedMessage.type:
        x = CRISPInitialisedMessage.decode_content(message.content)
        crisp_status['init'] = CRISPInitStatus.INITED if x else CRISPInitStatus.NOTINIT
    else:
        print(f"GUI: Unhandled message type: {message}")


async def connect_to_websocket():
    global websocket_connection, websocket_state
    while websocket_connection is None:
        try:
            websocket_connection = await connect("ws://localhost:8765")
            websocket_state['connected'] = True
            print("Connected to websocket")

            async def receive_messages():
                async for message in websocket_connection:
                    await on_message(message)

            asyncio.create_task(receive_messages())
        except Exception as e:
            print(f"Connection failed: {e}. Retrying in 1 second.")
            await asyncio.sleep(1)

    await websocket_connection.send(RequestConfigCameraMessage().encode())

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
    await websocket_connection.send(SetLEDMessage(content=(led, brightness)).encode())

async def request_crisp_status():
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket!")
        return
    await websocket_connection.send(CheckCRISPMessage().encode())

async def request_lock_crisp(lock):
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket")
        return
    await websocket_connection.send(SetCRISPMessage(lock).encode())

async def request_init_crisp():
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket")
        return
    crisp_status['init'] = CRISPInitStatus.INITING
    await websocket_connection.send(InitCRISPMessage().encode())

@ui.page('/')
async def page():
    await camera_config_event.wait() # Make sure the camera config is received before rendering the page

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
                with ui.column():
                    ui.button("Check CRISP", on_click=request_crisp_status)
                with ui.column():
                    ui.button("Lock CRISP", on_click=lambda: request_lock_crisp(True))
                with ui.column():
                    ui.button("Init CRSIP", on_click=lambda: request_init_crisp())
                with ui.column():
                    ui.button("Unlock CRISP", on_click=lambda: request_lock_crisp(False))
    #     ui.label("Crisp status")
    #     ui.label(str(crisp_status))
    #     ui.button("Init")

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