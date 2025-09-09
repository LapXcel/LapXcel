import socket
import json
import numpy as np

from crossq.utils.logx import colorize


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
        self.sock.connect((host, port))
        self.host = host
        self.port = port
        print(colorize(f"[Controller] Initialized TCP socket targeting ({self.host}, {self.port})", "cyan"))
    
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
            self.sock.sendto(msg, (self.host, self.port))
            print(colorize(f"[Controller] Sent {cmd_clean} to {(self.host, self.port)}", "cyan"))
        except Exception as e:
            print(colorize(f"[Controller] No data sent to client: {e}", "red"))
            self.on_close()
    
    def on_close(self) -> None:
        """
        Ensure socket is properly closed before terminating program.
        """
        print(colorize("[Controller] Closing socket connection", "cyan"))
        self.sock.close()

