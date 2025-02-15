#!/usr/bin/env python

import datetime
import asyncio
from websockets.asyncio.server import serve
import socket
import signal

from multiprocessing import Event, Queue
from launch_automaton import launch_automaton

SHUTDOWN_EXISTING = True

start_strategy_event: Event = None
stop_strategy_event: Event  = None
stop_event: Event = None
shutdown_event: Event = None
process_queue: Queue  = None
gui_to_automaton_queue: Queue = None
automaton_to_gui_queue: Queue = None

async def handler(websocket):
    global automaton_to_gui_queue, gui_to_automaton_queue, shutdown_event

    async def send_periodic_message():
        while True:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await websocket.send(f"Periodic message at time {current_time}")
            await asyncio.sleep(1)

    asyncio.create_task(send_periodic_message())


    # Receive messages
    async for message in websocket:
        await websocket.send(message)


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
        gui_to_automaton_queue,
        automaton_to_gui_queue
    ) = launch_automaton()

    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass