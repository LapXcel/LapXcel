from ac_controller import ACController
import socket
import json

def main():
    """
    The main function of the controller socket.
    This allows the model to put inputs into the game.
    """
    host = "127.0.0.1"
    port = 9999
    controller = ACController()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(1)
    print("[Controller] Socket listening on (" +
        str(host) + ", " + str(port) + ")")

    conn, addr = sock.accept()
    print(f"[Controller] Connection accepted from {addr}")

    while True:
        try:
            data = conn.recv(1024)
            if not data:
                break
            cmd = json.loads(data.decode())
            print(cmd)

            if cmd["perform"]:
                controller.perform(cmd["perform"][0], cmd["perform"][1])
                print(f"[Controller] Socket perform({cmd["perform"][0]}, {cmd["perform"][1]})")
            
            elif cmd["reset_car"]:
                controller.reset_car()
                print("[Controller] Socket reset car")
        except Exception as e:
            print(f"[Controller] Error: {e}")

    conn.close()
    sock.close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()