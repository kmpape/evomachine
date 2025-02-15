import nicegui.ui as ui
import nicegui.app as app

import asyncio
from websockets.asyncio.client import connect
from websockets import ClientConnection

RELOAD = True

websocket_connection: ClientConnection = None
websocket_state = {
    'connected': False
}

async def on_message(message):
    print(f"Received message: {message}")

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

async def hello():
    global websocket_connection
    if websocket_connection is None:
        print("Not connected to websocket!")
        return
    await websocket_connection.send("Hello world!")


def main():
    global websocket_connection

    # If reload is enabled, then it has to run the main program as a subprocess (https://github.com/zauberzeug/nicegui/issues/794)
    # => it runs the file again as a subprocess causing everything to be executed twice
    # => This reload runner doesn't need to run the actual code
    if RELOAD and __name__ == '__main__':
        print("Starting UI in subprocess")
        ui.run(host='127.0.0.1', port=8080, reload=RELOAD)
        return
    
    print("Building UI")
    ui.label("Say hello button: ")
    ui.button('Say Hello', on_click=hello)

    app.on_startup(connect_to_websocket)
    ui.run(host='127.0.0.1', port=8080, reload=RELOAD)


if __name__ in ['__main__', '__mp_main__']:
    main()