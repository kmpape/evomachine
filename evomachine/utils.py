import numpy as np
import pandas as pd
import time
from typing import Dict, List

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)


class Timer:
    def __init__(self, timer_level: int = 0, name: str = "", enabled: bool = True):
        self.timer_level: int = timer_level
        self.name: str = name
        self.enabled: bool = enabled
        self.timings: Dict[str, List] = {}

    def start(self, func_name: str, level: int) -> None:
        if (not self.enabled) or (level > self.timer_level):
            return
        if not (func_name in self.timings):
            self.timings[func_name] = []
        start_time = time.perf_counter()
        self.timings[func_name].append([start_time, None])

    def stop(self, func_name: str, level: int) -> None:
        if (not self.enabled) or (level > self.timer_level):
            return
        end_time = time.perf_counter()
        self.timings[func_name][-1][1] = end_time

    def get_timings(self) -> Dict[str, Dict]:
        timings = {}
        for func_name, start_end in self.timings.items():
            n_calls = len(start_end)
            elapsed = [end-start for start, end in start_end]
            timings[func_name] = {'n_calls': n_calls, 'avg': sum(elapsed)/n_calls, 'median': np.median(elapsed),
                                  'min': min(elapsed), 'max': max(elapsed)}
        return timings

    def get_timings_per_call(self) -> Dict[str, Dict]:
        timings = {}
        for func_name, start_end in self.timings.items():
            n_calls = len(start_end)
            elapsed = [end-start for start, end in start_end]
            timings[func_name] = elapsed
        return timings

    def display_timings(self) -> None:
        if not self.enabled:
            return
        print(f"\nTimings {self.name} (timer_level {self.timer_level}):\n")
        data = [[func_name, _data['n_calls'], _data['avg'], _data['median'], _data['min'], _data['max']]
                for func_name, _data in self.get_timings().items()]
        headers = ["name", "n_calls", "avg", "median", "min", "max"]
        df = pd.DataFrame(data, columns=headers)
        print(df)
        print("---\n")
