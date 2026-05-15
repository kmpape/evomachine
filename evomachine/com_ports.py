import re
import serial.tools.list_ports
from serial.tools.list_ports_common import ListPortInfo

def get_pid(in_str):
    match = re.search(r'VID:PID=(\w+:\w+)', in_str)
    if match:
        return match.group(1)
    else:
        return None

def listPorts() -> str:
    ports: list[ListPortInfo] = list(serial.tools.list_ports.comports())
    for port in ports:
        print("="*20)
        print("Device", port.device)
        print("Description", port.description)
        print("hwid", port.hwid)
        print("Serial number", port.serial_number)
        print("Location", port.location)
        print("Manufacturer", port.manufacturer)
        print("Product", port.product)

def get_port(hwid: str, display_name: str = ""):
    ports: list[ListPortInfo] = [port for port in list(serial.tools.list_ports.comports()) if port.hwid == hwid]
    if not ports:
        msg = f"No port with hwid {hwid} found for {display_name}."
        raise RuntimeError(msg)
    if len(ports) > 1:
        ports_str = "\n | ".join([str(port) for port in ports])
        msg = f"Multiple ports with hwid {hwid} found for {display_name}: {ports_str}"
        raise RuntimeError(msg)
    return ports[0].device

def get_syncboard_port():
    return get_port(hwid="16C0:0483", display_name="Syncboard")

def get_asitiger_port():
    return get_port(hwid="10C4:EA60", display_name="ASITiger")

def get_nvpro_port():
    return get_port(hwid="0483:A3E7", display_name="NVPro")

def get_kwr103_port():
    return get_port(hwid="16C0:0483", display_name="KWR103")

if __name__ == "__main__":
    listPorts()
    print("-"*20)
    print("Syncboard port:", get_syncboard_port())
    print("ASITiger port:", get_asitiger_port())
