import socket

from crossq.utils.logx import colorize


class ACSocket:
    """
    Socket connection with the Assetto Corsa app.
    This is used to get real-time data from the game and send it to the RL model.
    """

    sock = None  # Socket object for the server
    conn = None  # Connection object for the client connection
    addr = None  # Address of the connected client
    data = None  # Variable to store received data from the client

    def __init__(self, host: str = "127.0.0.1", port: int = 65431) -> None:
        """
        Set up the socket connection.
        :param host: The host to connect to (default: localhost)
        :param port: The port to connect to (default: 65431)
        """
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow the socket to reuse the address
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to the specified host and port
        self.sock.bind((host, port))
        # Listen for incoming connections (0 means no backlog)
        self.sock.listen(0)
        print(colorize("[ACRL] Socket listening on (" +
              str(host) + ", " + str(port) + ")", "cyan"))

    def connect(self) -> socket:
        """
        Wait for an incoming connection and return the socket object.
        """
        # Accept an incoming connection and store the connection and address
        self.conn, self.addr = self.sock.accept()
        print(colorize("[ACRL] Connected by " + str(self.addr), "cyan"))
        return self.conn  # Return the connection object for further communication

    def update(self) -> None:
        """
        Send a message to the client to request data, and then receive the data.
        """
        try:
            # Send a request for the next state of the game
            self.conn.sendall(b"next_state")
            # Uncomment the following line to print a debug message
            # print("[ACRL] Sent data request to client")
            # Receive data from the client (up to 1024 bytes)
            self.data = self.conn.recv(1024)
            # Uncomment the following line to print a debug message
            # print("[ACRL] Received data from client")
        except:
            # Handle exceptions (e.g., if the client disconnects)
            print(colorize(
                "[ACRL] No data received from client, closing socket connection", "red"))
            self.on_close()  # Close the socket connection on error

    def end_training(self) -> None:
        """
        Send an empty message to the client so it knows training has been completed.
        """
        try:
            # Send an empty message to indicate training is complete
            self.conn.sendall(b"")
            print(
                colorize("[ACRL] Sent training completed message to client", "green"))
        except:
            # Handle exceptions (e.g., if the client disconnects)
            print(
                colorize("[ACRL] No response from client, closing socket connection", "red"))
            self.on_close()  # Close the socket connection on error

    def on_close(self) -> None:
        """
        Ensure socket is properly closed before terminating program.
        """
        print(colorize("[ACRL] Closing socket connection", "cyan"))
        self.sock.close()  # Close the server socket
