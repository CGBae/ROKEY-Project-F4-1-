import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rokey/Desktop/rokey_F4/fuel_port_perception/install/fuel_port_perception'
