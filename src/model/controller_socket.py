import socket
import json
import numpy as np

from utils.logx import colorize


class ControllerSocket:
    """
    Socket connection with the Assetto Corsa app.
    This is used to get real-time data from the game and send it to the RL model.
    """

    sock = None
    conn = None
    addr = None
    data = None

class ControllerSocket:
    def __init__(self, host="host.docker.internal", port=9999) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.host = host
        self.port = port
        self.sock.connect((self.host, self.port))
    
    def convert_np(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        if isinstance(obj, dict):
            return {k: self.convert_np(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_np(i) for i in obj]
        return obj

    def update(self, command) -> None:
        try:
            cmd_clean = self.convert_np(command)
            msg = json.dumps(cmd_clean).encode()
            self.sock.sendall(msg)
            # print(colorize(f"[Controller] Sent {cmd_clean}", "cyan"))

        except Exception as e:
            print(colorize(f"[Controller] Error: {e}", "red"))
            self.on_close()
    
    def on_close(self) -> None:
        """
        Ensure socket is properly closed before terminating program.
        """
        print(colorize("[Controller] Closing socket connection", "cyan"))
        self.sock.close()

