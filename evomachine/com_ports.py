import re
import serial.tools.list_ports

ports = list(serial.tools.list_ports.comports())

def get_pid(in_str):
    match = re.search(r'VID:PID=(\w+:\w+)', in_str)
    if match:
        return match.group(1)
    else:
        return None

def listPorts():
    for port in ports:
        print("="*20)
        print("Device", port.device)
        print("Description", port.description)
        print("hwid", port.hwid)
        print("Serial number", port.serial_number)
        print("Location", port.location)
        print("Manufacturer", port.manufacturer)
        print("Product", port.product)

def get_syncboard_port():
    for port in ports:
        if get_pid(port.hwid) == "16C0:0483":
            return port.device
    raise RuntimeError("Syncboard not found")

def get_asitiger_port():
    for port in ports:
        if get_pid(port.hwid) == "10C4:EA60":
            return port.device
    raise RuntimeError("ASITiger not found")

def get_nvpro_port():
    for port in ports:
        if get_pid(port.hwid) == "0483:A3E7":
            return port.device
    raise RuntimeError("NVPro not found")

if __name__ == "__main__":
    listPorts()
    print("-"*20)
    print("Syncboard port:", get_syncboard_port())
    print("ASITiger port:", get_asitiger_port())