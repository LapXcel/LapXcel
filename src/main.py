from ac_controller import ACController
from ac_socket import ACSocket

def main():
    """
    Main Function
    """
    controller = ACController()
    sock = ACSocket()

    with sock.connect() as conn:
        while(1):
            sock.update()
            print(sock.data)
            controller.perform(1, 0)

if __name__ == "__main__":
    """
    Run the main function.
    """
    main()