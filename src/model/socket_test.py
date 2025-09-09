from ac_socket import ACSocket

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Establish a socket connection
    sock = ACSocket()
    print("Waiting for socket connection...")
    sock.connect()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()