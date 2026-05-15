# some code from https://github.com/mbrennwa/pypsucurvetrace/tree/master

import serial
import time


class KWR103:

    def __init__(self, port: str):
        self.port: str = port
        self._Serial: serial.Serial | None = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def _query(self, cmd, reply=False):
        if self._Serial is None:
            raise RuntimeError("KWR103 is not connected.")

        self._Serial.reset_output_buffer()
        self._Serial.reset_input_buffer()
        time.sleep(0.03)

        self._Serial.write((cmd+'\n').encode())

        if not reply:
            ans = None
        else:
            ans = self._Serial.readline().decode('utf-8').rstrip("\n\r")
            if ans == '':
                raise RuntimeError("No replay from KWR103")

        return ans
    
    def connect(self):
        self._Serial = serial.Serial(self.port, baudrate=115200)

    def is_connected(self) -> bool:
        return bool(self._Serial is not None and self._Serial.is_open)

    def disconnect(self):
        self.close()

    def close(self):
        if self._Serial is not None and self._Serial.is_open:
            self._Serial.close()

    def query_serial_no(self):
        return self._query('*IDN?', reply=True)
    
    def set_output(self, status=True):
        self._query(f'OUT:{1 if status else 0}')

    def set_voltage(self, val):
        self._query(f'VSET:{val:2.1f}')

    def set_current(self, val):
        self._query(f'ISET:{val:2.1f}')

    def get_voltage_set(self):
        return float(self._query('VSET?', reply=True))
    
    def get_current_set(self):
        return float(self._query('ISET?', reply=True))
    
    def get_voltage_out(self):
        return float(self._query('VOUT?', reply=True))
    
    def get_current_out(self):
        return float(self._query('IOUT?', reply=True))
