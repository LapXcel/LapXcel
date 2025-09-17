from ac_controller import ACController
import socket
import mss
import numpy as np
import cv2
import base64
import json

def main():
    """
    The main function of the controller socket.
    This allows the model to put inputs into the game.
    """
    controller = ACController()
    ac_host = "127.0.0.1"
    ac_port = 65431
    ac_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ac_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ac_sock.bind((ac_host, ac_port))
    ac_sock.listen(1)
    print("[AC] Socket listening on (" +
        str(ac_host) + ", " + str(ac_port) + ")")
    ac_conn, ac_addr = ac_sock.accept()
    
    host = "127.0.0.1"
    port = 9999
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.connect((host, port))
    print(f"[Controller] Connected to ({host}, {port})")

    while True:
        with mss.mss() as sct:
            mon = sct.monitors[2]
            controller = ACController()
            while True:
                # Receive signals from the socket
                raw_msg = sock.recv(1024)
                if not raw_msg:
                    print("[Controller] Ending training...")
                    break
                message_str = raw_msg.decode('utf-8')
                message = json.loads(message_str)

                print(f"[Controller] Received request: {message}")

                if 'msg_type' not in message:
                    print('Expected msg_type field')
                    continue

                if message["msg_type"] == "request":
                    ac_conn.update()
                    data = ac_conn.data

                    try:
                        # Convert the byte data to a string
                        data_str = data.decode('utf-8')

                        # Split the string by commas and map values to a dictionary
                        data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
                    except:
                        # If the data is invalid, throw error and return empty dict
                        print(("Error parsing data, returning empty dict!"))
                        continue

                    screenshot = sct.grab(mon)
                    img = np.array(screenshot)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, encoded_img = cv2.imencode('.jpg', img)
                    img_bytes = encoded_img.tobytes()
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

                    cv2.imshow('Image Window', img)

                    data = {"msg_type": "telemetry", "image": img_b64, "lap_invalid": data_dict["lap_invalid"], "speed": data_dict["speed_kmh"], "steering_angle": data_dict["steer"]}
                    print(f"[Controller] Sending data: {data}")

                    sock.sendall(data)

                elif message["msg_type"] == "control":
                    print("[Controller] Performing action...")
                    controller.perform(message["throttle_brake"], message["steer"])

                elif message["msg_type"] == "reset_car":
                    print("[Controller] Resetting...")
                    controller.reset_car()
                
                else:
                    print("[Controller] Error: Invalid Message Type")
                    continue

        ac_conn.close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()