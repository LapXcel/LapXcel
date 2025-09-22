# Original author: Tawn Kramer
# Edited by: Colby Todd

import asyncio
import base64
import time
from io import BytesIO
from threading import Thread
import time

import numpy as np
from PIL import Image

from config import INPUT_DIM, ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, \
    REWARD_CRASH, CRASH_SPEED_WEIGHT
from envs.utils.fps import FPSTimer
from envs.utils.tcp_server import IMesgHandler, SimServer


class ACController:
    """
    Wrapper for communicating with unity simulation.

    :param port: (int) Port to use for communicating with the simulator
    """

    def __init__(self, port=9999, verbose=False):
        self.verbose = verbose

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ('0.0.0.0', port)

        # Socket message handler
        self.handler = ACHandler(self.verbose)
        # Create the server to which the unity sim will connect
        self.server = SimServer(self.address, self.handler, self.verbose)
        # Create a new event loop for the thread
        self.loop = asyncio.new_event_loop()

        self.thread = Thread(target=self.start_loop, args=(self.loop,))
        self.thread.daemon = True
        self.thread.start()
    
    async def start_server(self):
        await self.server.start()

    def start_loop(self, loop):
        if self.verbose:
            print("[ACController] Starting asyncio event loop and server")
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server.start())
        if self.verbose:
            print("[ACController] Server started, running event loop")
        loop.run_forever()


    def close_connection(self):
        return self.server.handle_close()

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        while not self.handler.loaded:
            print("Waiting for sim to start..."
                  "if the simulation is running, press EXIT to go back to the menu")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class ACHandler(IMesgHandler):
    """
    Socket message handler.

    """

    def __init__(self, verbose):
        self.sock = None
        self.loaded = False
        self.verbose = verbose
        self.timer = FPSTimer(verbose=0)

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        self.steering_angle = 0.0
        self.current_step = 0
        self.last_step = -1
        self.speed = 0
        self.steering = None
        self.lap_invalid = False
        self.slow = -1
        self.track_progress = 0
        self.last_track_progress = 0
        # Define which method should be called
        # for each type of message
        self.fns = {'telemetry': self.on_telemetry}

    def on_connect(self, socket_handler):
        """
        :param socket_handler: (socket object)
        """
        if self.verbose:
            print("[ACHandler] onconnect called - socket connection established")
        self.sock = socket_handler
        self.loaded = True

    def on_disconnect(self):
        """
        Close socket.
        """
        self.sock.connection_lost(None)
        self.sock = None

    def on_recv_message(self, message):
        """
        Distribute the received message to the appropriate function.

        :param message: (dict)
        """
        if 'msg_type' not in message:
            print('[ACHandler] Expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('[ACHandler] Unknown message type', msg_type)

    def reset(self):
        """
        Global reset, notably it
        resets car to initial position.
        """
        if self.verbose:
            print("[ACHandler] resetting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.current_step = 0
        self.last_step = -1
        self.lap_invalid = False
        self.send_control(0, 0)
        self.send_reset_car()
        time.sleep(1.0)
        self.timer.reset()
        self.slow = -1

    def get_sensor_size(self):
        """
        :return: (tuple)
        """
        return self.camera_img_size

    def take_action(self, action):
        """
        :param action: ([float]) Steering and throttle
        """
        if self.verbose:
            print("[ACHandler] take_action")

        throttle = action[1]
        self.steering = action[0]
        self.last_throttle = throttle
        self.current_step += 1

        self.send_control(self.steering, throttle)

    def observe(self):
        msg = {'msg_type': 'request'}
        self.queue_message(msg)

        while self.last_obs is self.image_array:
            time.sleep(1.0/60.0)
        
        self.last_step = self.current_step
        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        truncated = time.time() - self.slow >= 5 and self.slow != -1
        reward = self.calc_reward(done, truncated)
        info = {}

        self.timer.on_frame()

        return observation, reward, done, truncated, info

    def is_game_over(self):
        """
        :return: (bool)
        """
        # if self.steps_since_reset < 2:
        #     return False
        if self.verbose:
            print(f"[ACHandler] Is Game Over {self.lap_invalid}")
        return self.lap_invalid

    def calc_reward(self, done, truncated):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash

        :param done: (bool)
        :return: (float)
        """
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        if truncated:
            return -5
        # 1 per timesteps + throttle
        # throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        reward = self.track_progress - self.last_track_progress
        return reward

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        """
        Update car info when receiving telemetry message.

        :param data: (dict)
        """
        img_string = data["image"]
        if self.verbose:
            print(f"[ACHandler] Image base64 length: {len(img_string)}")
        try:
            image = Image.open(BytesIO(base64.b64decode(img_string)))
        except Exception as e:
            print(f"[ACHandler] Failed to open image: {e}")
            return
        image = np.array(image)
        # Save original image for render
        # self.original_image = np.copy(image)
        # Resize if using higher resolution images
        # image = cv2.resize(image, CAMERA_RESOLUTION)
        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Convert RGB to BGR
        # image = image[:, :, ::-1]
        self.image_array = image
        # Here resize is not useful for now (the image have already the right dimension)
        # self.image_array = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        self.lap_invalid = data["lap_invalid"] == "True"
        self.last_track_progress = self.track_progress
        self.track_progress = float(data["track_progress"]) * 100
        self.steering_angle = float(data['steering_angle'])
        self.speed = float(data["speed"])
        if self.speed >= 5:
            self.slow = -1
        elif self.speed < 5 and self.slow == -1:
            self.slow = time.time()


    def send_control(self, steer, throttle):
        """
        Send message to the server for controlling the car.

        :param steer: (float)
        :param throttle: (float)
        """
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__()}
        self.queue_message(msg)

    def send_reset_car(self):
        """
        Reset car to initial position.
        """
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def queue_message(self, msg):
        """
        Add message to socket queue.

        :param msg: (dict)
        """
        if self.sock is None:
            if self.verbose:
                print('[ACHandler] skipping:', msg)
            return

        if self.verbose:
            print('[ACHandler] sending', msg)
        self.sock.queue_message(msg)