#!/usr/bin/env python

import sys
import os
import asyncio
from websockets.asyncio.server import serve
from websockets import WebSocketServerProtocol
import socket
import signal

from multiprocessing import Event, Queue
from launch_automaton import launch_automaton, QueueManager, ConfigCamera, LEDType

from messages import Message, MessageType

SHUTDOWN_EXISTING = True

start_strategy_event: Event = None
stop_strategy_event: Event  = None
stop_event: Event = None
shutdown_event: Event = None
process_queue: Queue  = None
gui_to_automaton_queue: Queue = None
automaton_to_gui_queue: Queue = None
queue_manager: QueueManager = None
camera_config: ConfigCamera = None

def set_led(led, brightness):
    global queue_manager
    queue_manager.request(
        req_str='self.cam.set_led',
        kwargs_dict={'i_chan': led, 'brightness': brightness},
        # callback=self.update_led,
        # callback_args=(i_channel, brightness,),
    )

async def receive_crisp_status(status, websocket: WebSocketServerProtocol):
    print("Received CRISP status:", status)
    await websocket.send(Message(content=status, type=MessageType.crisp_status).encode())

    import numpy as np
    await websocket.send(Message(content=(np.random.rand(360, 640, 3) * 255).astype('uint8'), type=MessageType.image).encode())

async def receive_crisp_initialised(status, websocket: WebSocketServerProtocol):
    print("Received CRISP initialised:", status)
    await websocket.send(Message(content=status, type=MessageType.crisp_initialised).encode())

# This shows how to register an async callback
def check_crisp_status(websocket, loop):
    global queue_manager
    queue_manager.request(
        req_str='self.cam.autofocus_is_locked',
        kwargs_dict={},
        callback=lambda status: asyncio.run_coroutine_threadsafe(receive_crisp_status(status, websocket), loop),
    )
    queue_manager.request(
        req_str='self.cam.autofocus_is_initialised',
        kwargs_dict={},
        callback=lambda status: asyncio.run_coroutine_threadsafe(receive_crisp_initialised(status, websocket), loop),
    )

# This shows how to register a normal callback
def set_crisp(lock: bool, websocket, loop):
    global queue_manager
    queue_manager.request(
        req_str='self.cam.autofocus_' + ('lock' if lock else 'unlock'),
        kwargs_dict={},
        callback=lambda _: check_crisp_status(websocket, loop),
    )

def init_crisp(cfg_crisp, websocket, loop):
    global queue_manager
    queue_manager.request(
        req_str='self.cam.autofocus_initialise',
        kwargs_dict={'this_cfg_crisp': cfg_crisp, 'user_input': False},
        callback=lambda _: check_crisp_status(websocket, loop),
    )

def request_frame(websocket, loop):
    global queue_manager
    queue_manager.request(
        req_str='self.cam.get_frame',
        kwargs_dict={'i_chan': None},
        callback=lambda frame: asyncio.run_coroutine_threadsafe(websocket.send(Message(content=frame, type=MessageType.image).encode()), loop),
    )

async def handler(websocket):
    global automaton_to_gui_queue, gui_to_automaton_queue, shutdown_event, camera_config

    loop = asyncio.get_event_loop()

    async for message in websocket:
        
        # Attempt to decode the message
        try:
            message = Message.decode(message)
        except Exception as e:
            print(f"Failed to decode message: {e}")
            continue
        
        # await websocket.send(Message(content=f"Server received message of type: {message.type}", type=MessageType.text).encode())

        if message.type not in MessageType:
            print(f"Microscope Server: Received unknown message type: {message.type}")
            continue

        if message.type == MessageType.set_led:
            led, brightness = message.content
            set_led(led, brightness)
        elif message.type == MessageType.request_camera_config:
            await websocket.send(Message(content=camera_config, type=MessageType.camera_config).encode())
        elif message.type == MessageType.check_crisp_status:
            check_crisp_status(websocket, loop)
        elif message.type == MessageType.set_crisp:
            new_crisp_lock = message.content
            set_crisp(new_crisp_lock, websocket, loop)
        elif message.type == MessageType.init_crisp:
            init_crisp(cfg_crisp=camera_config.autofocus, websocket=websocket, loop=loop)
        elif message.type == MessageType.request_frame:
            request_frame(websocket, loop)
        else:
            print(f"Microscope Server: Received unhandled message type: {message.type}")

async def main():
    async with serve(handler, "localhost", 8765) as server:
        await server.serve_forever()

async def shutdown(loop):
    global shutdown_event

    shutdown_event.set()

    print("Shutting down server...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def is_server_running(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, lambda s, f: asyncio.ensure_future(shutdown(loop)))

    if is_server_running("localhost", 8765):
        print("Server is already running.")
        if SHUTDOWN_EXISTING:
            print("Shutting down the already running server.")
            os.system("fuser -k 8765/tcp")
        else:
            sys.exit(1)

    print("I only run once!")

    (
        start_strategy_event,
        stop_strategy_event,
        stop_event,
        shutdown_event,
        process_queue,
        queue_manager,
        gui_to_automaton_queue,
        automaton_to_gui_queue,
        camera_config
    ) = launch_automaton()

    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass