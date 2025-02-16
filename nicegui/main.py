import nicegui.ui as ui
import nicegui.app as app

import asyncio
from websockets.asyncio.client import connect
from websockets import ClientConnection

from messages import Message, TextMessage, SetLEDMessage, RequestConfigCameraMessage, CameraConfigMessage
from evomachine.evotypes import LEDType

RELOAD = True

websocket_connection: ClientConnection = None
websocket_state = {
    'connected': False
}

led_brightness = {}

camera_config = None
camera_config_event = asyncio.Event()

async def on_message(message):
    global camera_config
    message = Message.decode(message)
    print(f"Received message type: {message}")

    if message.type == 'cameraconfig':
        camera_config = CameraConfigMessage.decode_content(message.content).content
        camera_config_event.set()


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

@ui.page('/')
async def page():
    await camera_config_event.wait() # Make sure the camera config is received before rendering the page

    with ui.card():
        for led in camera_config.leds:
            with ui.row():
                led_brightness.setdefault(led.name, 29.0)
                ui.label(f"LED {led.name}")
                ui.number(value=led_brightness[led.name], min=0, max=100, step=1).bind_value(led_brightness, led.name).on(type='keydown.enter', handler=lambda led=led: set_led(led, led_brightness[led.name]))
                ui.button("Set LED", on_click=lambda led=led: set_led(led, led_brightness[led.name]))

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
        ui.run(host='127.0.0.1', port=8080, reload=RELOAD)
        return

    app.on_startup(connect_to_websocket)
    app.on_shutdown(disconnect_websocket)

    ui.run(host='127.0.0.1', port=8080, reload=RELOAD)

if __name__ in ['__main__', '__mp_main__']:
    main()