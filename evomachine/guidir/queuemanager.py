from multiprocessing import Event, Lock, Queue
import queue
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from evomachine.commands import AutomatonCommand
from evomachine.config import get_logger
from evomachine.evotypes import AutomatonCommandType


logger = get_logger(name=__name__, is_gui=True)


class QueueManager:
    def __init__(
            self,
            process_q: Queue,
            gui_to_automaton_q: Queue,
            automaton_to_gui_q: Queue,
            start_strategy_event: Event,
            stop_event: Event,
            shutdown_event: Event,
            request_lock: Optional[Lock] = None,
            queue_timeout: float = 0,
            run_timeout: float = 0,
            use_threading: bool = True, 

    ):
        self._process_q: Queue = process_q
        "Filled by Automaton process. Format: (AutomatonCommandType, AutomatonCommand)"
        self._gui_to_automaton_q: Queue = gui_to_automaton_q
        "Filled by QueueManager. Format: (cmd_id: UUID, cmd_str: str, kwargs_dict: Dict[str, Any])"
        self._automaton_to_gui_q: Queue = automaton_to_gui_q
        "Filled by Automaton. Format: (cmd_id: UUID, data: Any)"
        self._request_lock: Optional[Lock] = request_lock
        "Locks _requests if not None."
        self._strategy_event: Event = start_strategy_event
        "Only used to check for bad gui-automaton communication."
        self._stop_event: Event = stop_event
        "Stops the event loop."
        self._shutdown_event: Event = shutdown_event
        "Shuts down process."
        self._listeners: Dict[AutomatonCommandType, List[Callable[[AutomatonCommand], None]]] = {
            key: [] for key in AutomatonCommandType.get_all()
        }
        "Listeners (callbacks) for AutomatonCommandType messages during process."
        self._requests: Dict[uuid.UUID, Tuple[Callable[[Any, Optional], None], Tuple[Any]]] = {}
        "Request (callbacks) from GUI to Automaton."
        self.queue_timeout: float = queue_timeout
        "Timeout for polling all queues."
        self.run_timeout: float = run_timeout
        "Timeout after each iteration."
        self.use_threading: bool = use_threading
        "If True, spawns a new thread for each callback."

    def _answer(self, request_id: uuid.UUID, data: Any):
        logger.debug(f"Answering request {request_id} with response {data}")
        if request_id not in self._requests:
            logger.warning(f"Received unexpected response: {request_id}")
            raise RuntimeError(f"{request_id} not in {self._requests}")
        if self._request_lock is not None:
            logger.debug(f"Locking _requests for answer {request_id} with response {data} and callback"
                         f" {self._requests[request_id][0].__qualname__ if self._requests[request_id][0] else None}")
            if self._requests[request_id][0] is not None:
                if self.use_threading:
                    if self._requests[request_id][1] is None:
                        threading.Thread(target=self._requests[request_id][0], args=(data,)).start()
                    else:
                        threading.Thread(target=self._requests[request_id][0],
                                         args=(data, *self._requests[request_id][1])).start()
                else:
                    if self._requests[request_id][1] is None:
                        self._requests[request_id][0](data)
                    else:
                        self._requests[request_id][0](data, *self._requests[request_id][1])
            self._request_lock.acquire()
            try:
                del self._requests[request_id]
            finally:
                self._request_lock.release()
        else:
            if self._requests[request_id][0] is not None:
                if self.use_threading:
                    if self._requests[request_id][1] is None:
                        threading.Thread(target=self._requests[request_id][0], args=(data,)).start()
                    else:
                        threading.Thread(target=self._requests[request_id][0],
                                         args=(data, *self._requests[request_id][1])).start()
                else:
                    if self._requests[request_id][1] is None:
                        self._requests[request_id][0](data)
                    else:
                        self._requests[request_id][0](data, *self._requests[request_id][1])
            del self._requests[request_id]

    def has_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def register(self, func: Callable[[AutomatonCommand], None], msg_type: AutomatonCommandType):
        logger.debug(f"Registering {func} for {msg_type} messages")
        self._listeners[msg_type].append(func)

    def request(
            self,
            req_str: str,
            kwargs_dict: Dict[str, Any],
            callback: Optional[Callable[[Any], None]] = None,
            callback_args: Optional[Tuple] = None,
    ):
        # if 'Automaton' not in request_func.__qualname__:
        #     logger.warning(f"Request function {request_func} does not belong to Automaton.")
        #     raise RuntimeError(f"Request function {request_func} does not belong to Automaton.")
        # req_str = request_func.__qualname__.replace('Automaton.', 'self.')
        req_id = uuid.uuid4()
        logger.debug(f"Received request {req_id} with {req_str}, {kwargs_dict} and callback "
                     f"{callback.__qualname__ if callback else None}")
        if self.strategy_started():
            logger.warning(f"Received request {req_id} with {req_str} and {kwargs_dict} while strategy is running.")
            raise RuntimeError(f"Received request {req_id} with {req_str} and {kwargs_dict} while strategy is running.")
        if self._request_lock is not None:
            logger.debug(f"Locking _requests for {req_id} with {req_str}, {kwargs_dict} and callback "
                         f"{callback.__qualname__ if callback else None}")
            self._request_lock.acquire()
            try:
                self._requests[req_id] = (callback, callback_args)
            finally:
                self._request_lock.release()
        else:
            self._requests[req_id] = (callback, callback_args)
        self._gui_to_automaton_q.put((req_id, req_str, kwargs_dict))

    def restart(self):
        logger.debug('Restarting QueueManager')
        if not self.stopped():
            logger.warning('QueueManager is already running')
        self._stop_event.clear()

    def run(self):
        while not self.has_shutdown():
            while not self.stopped():
                # logger.debug("Polling _automaton_to_gui_q")
                while not self._automaton_to_gui_q.empty():
                    try:
                        request_id, data = self._automaton_to_gui_q.get(block=True, timeout=self.queue_timeout)
                        if isinstance(data, Exception):
                            logger.error(f"Received error: {data} for request {request_id}")
                        logger.debug(f"QueueManager received {request_id} from _automaton_to_gui_q")
                        if self.strategy_started():
                            logger.warning("Received response while strategy is running.")
                            raise RuntimeError("Received response while strategy is running.")
                        self._answer(request_id=request_id, data=data)
                    except queue.Empty:
                        pass
                # logger.debug("Polling _process_q")
                while not self._process_q.empty():
                    try:
                        # Note: This queue is also filled when strategy is NOT running.
                        queue_type, data = self._process_q.get(block=True, timeout=self.queue_timeout)
                        if isinstance(data, Exception):
                            logger.error(f"Received error: {data} for request {request_id}")
                        logger.debug(f"QueueManager received {queue_type} from _process_q")
                        for func in self._listeners[queue_type]:
                            logger.debug(f"Calling {func} with {data}")
                            if self.use_threading:
                                threading.Thread(target=func, args=(data,)).start()
                            else:
                                func(data)
                    except queue.Empty:
                        pass
                if self.run_timeout > 0:
                    self.sleep(duration=self.run_timeout)

    def shutdown(self):
        logger.debug('Shutting Down QueueManager')
        self._shutdown_event.set()

    def sleep(self, duration: float):
        now = time.perf_counter()
        end = now + duration
        while (now < end) and not self.stopped():
            now = time.perf_counter()

    def stop(self):
        logger.debug('Stopping QueueManager')
        self._stop_event.set()

    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def strategy_started(self) -> bool:
        return self._strategy_event.is_set()

# class QueueManager(threading.Thread):
#     def __init__(self, data_queue: queue.Queue, queue_timeout: float = 0):
#         super().__init__()
#         self.main_queue: queue.Queue = data_queue
#         self._stop_event = threading.Event()
#         self._listeners: Dict[AutomatonCommandType, List[Callable[[AutomatonCommand], None]]] = {
#             key: [] for key in AutomatonCommandType.get_all()
#         }
#         self.queue_timeout = queue_timeout
#
#     def register(self, func: Callable[[AutomatonCommand], None], msg_type: AutomatonCommandType):
#         logger.debug(f"Registering {func} for {msg_type} messages")
#         self._listeners[msg_type].append(func)
#
#     def run(self) -> None:
#         while True:
#             while not self.stopped():
#                 try:
#                     # NOTE the command type in tmp_data[0] might differ from the one in tmp_data[1].command_type.
#                     # This is because PROCESS_DATA might include various command types.
#                     tmp_data = self.main_queue.get(block=True, timeout=self.queue_timeout)
#                     logger.debug(f"QueueManager received {tmp_data[0]}")
#                     for func in self._listeners[tmp_data[0]]:
#                         logger.debug(f"Calling {func} for {tmp_data[0]}")
#                         func(tmp_data[1])
#                 except queue.Empty:
#                     pass
#                 if self.queue_timeout:
#                     self.sleep(self.queue_timeout)
#
#     def stop(self):
#         logger.debug('Stopping QueueManager')
#         self._stop_event.set()
#
#     def stopped(self):
#         return self._stop_event.is_set()
#
#     def sleep(self, duration: float):
#         now = time.perf_counter()
#         end = now + duration
#         while (now < end) and not self.stopped():
#             now = time.perf_counter()
