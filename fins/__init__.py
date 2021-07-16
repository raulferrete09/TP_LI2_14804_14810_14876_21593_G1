from abc import ABCMeta, abstractmethod
from PyQt5.QtCore import QTimer, qVersion
import time
import signal
import sys
import numpy as nd


class FinsResponseEndCode:
    def __init__(self):
        self.NORMAL_COMPLETION = b'\x00\x00'
        self.SERVICE_CANCELLED = b'\x00\x01'


class FinsConnection(metaclass=ABCMeta):
    def __init__(self):
        self.dest_node_add = 0
        self.srce_node_add = 0
        self.dest_net_add = 0
        self.srce_net_add = 0
        self.dest_unit_add = 0
        self.srce_unit_add = 0

    def fins_command_frame(self, command_code, text=b'', service_id=b'\x60',
                           icf=b'\x80', gct=b'\x07', rsv=b'\x00'):
        command_bytes = icf + rsv + gct + \
                        self.dest_net_add.to_bytes(1, 'big') + self.dest_node_add.to_bytes(1, 'big') + \
                        self.dest_unit_add.to_bytes(1, 'big') + self.srce_net_add.to_bytes(1, 'big') + \
                        self.srce_node_add.to_bytes(1, 'big') + self.srce_unit_add.to_bytes(1, 'big') + \
                        service_id + command_code + text
        return command_bytes

    """  WRITE DATA ON PLC """

    def Init(self, rw):

        MEMORY_AREA_WRITE = b'\x01\x02'

        """  Initializate system"""
        if (rw == 99):
            data = b'\x82\x00\x02\x00\x00\x01\x00\x01'

        """  Operation on  ---- busy time '50' seconds for testing"""
        if (rw == 50):
            data = b'\x82\x00\x03\x00\x00\x01\x00\x09'

        """  Turn off all outputs on INIT"""
        if (rw == 10):
            data = b'\xB0\x00\x01\x00\x00\x01\x00\x00'
            data = b'\x82\x00\x01\x00\x00\x01\x00\x00'

        response = self.execute_fins_command_frame(
            self.fins_command_frame(MEMORY_AREA_WRITE, data))
        return response

    """  READ DATA ON PLC """

    def Read_Sate(self, rw):

        MEMORY_AREA_READ = b'\x01\x01'

        """  Read 2 first bytes in D memory"""
        if (rw == 20):
            data = b'\x82\x00\x00\x04\x00\x01'

        response = self.execute_fins_command_frame(self.fins_command_frame(MEMORY_AREA_READ, data))

        response = (response.hex()[28:32])
        return response



