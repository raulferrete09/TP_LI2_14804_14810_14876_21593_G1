import sys

from PyQt5.QtCore import QTimer, qVersion
import socket
from fins import FinsConnection
from threading import Timer

class Watchdog(Exception):
    def __init__(self, timeout, userHandler=None):  # timeout in seconds
        self.timeout = timeout
        self.handler = userHandler if userHandler is not None else self.defaultHandler
        self.timer = Timer(self.timeout, self.handler)
        self.timer.start()
    def reset(self):
        self.timer.cancel()
        self.timer = Timer(self.timeout, self.handler)
        self.timer.start()

    def stop(self):
        self.timer.cancel()

    def defaultHandler(self):
        raise self

class UDPFinsConnection(FinsConnection):

    """
    """
    def __init__(self):
        super().__init__()
        self.BUFFER_SIZE=4096
        self.fins_socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.ip_address='192.168.1.101'
        self.fins_port=9600

    def execute_fins_command_frame(self,fins_command_frame):
        """Sends FINS command """

        response = ""
        self.fins_socket.sendto(fins_command_frame,(self.ip_address,9600))

        try:
                response = self.fins_socket.recv(self.BUFFER_SIZE)
        except Exception as err:
            print(err)
        return response

    def connect(self, IP_Address, Port=9600, Bind_Port=9600):
        """Establish a connection for FINS communications

        :param IP_Address: The IP address of the device you are connecting to
        :param Port: The port that the device and host should listen on (default 9600)
        """
        self.fins_port=Port
        self.ip_address=IP_Address
        self.fins_socket.bind(('',Bind_Port))
        self.fins_socket.settimeout(1.0)

    def __del__(self):
        self.fins_socket.close()