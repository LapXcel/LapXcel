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
            try:
                controller.perform(0.5, 0)
                print("Controller input sent")
            except Exception as e:
                print(f"Error sending input: {e}")

if __name__ == "__main__":
    """
    Run the main function.
    """
    main()