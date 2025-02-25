import math
from typing import Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from ac_controller import ACController
from ac_socket import ACSocket
from crossq.utils.logx import colorize


class Env(gym.Env):
    """
    The custom gymnasium environment for the Assetto Corsa game.
    """

    metadata = {"render_modes": [], "render_fps": 0}
    _observations = None
    _invalid_flag = 0.0
    _sock = None

    def __init__(self, render_mode: Optional[str] = None, max_speed=200.0):
        # Initialize the controller
        self.controller = ACController()
        self.max_speed = max_speed

        # Observations is a Box with the following data:
        # - "track_progress": The progress of the car on the track, in [0.0, 1.0]
        # - "speed_kmh": The speed of the car in km/h [0.0, max_speed]
        # - "world_loc_x": The world's x location of the car [-2000.0, 2000.0]
        # - "world_loc_y": The world's y location of the car [-2000.0, 2000.0]
        # - "world_loc_z": The world's z location of the car [-2000.0, 2000.0]
        # - "lap_invalid": Whether the current lap is valid [0.0, 1.0]
        # - "lap_count": The current lap count [1.0, 2.0]
        # - "lap_time": The current time of the lap in ms [0, 120000]
        # - "previous_track_progress": The previous track progress [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=np.array(
                [0.000, 0.0, -2000.0, -2000.0, -2000.0, 0.0, 1.0, 0, 0]),
            high=np.array([1.000, max_speed, 2000.0,
                          2000.0, 2000.0, 1.0, 2.0, 120000, 500]),
            shape=(10,),
            dtype=np.float32,
        )

        # We have a continuous action space, where we have:
        # - A brake/throttle value, which is a number in [-1.0, 1.0]
        # - A steering angle, which is a number in [-1.000, 1.000]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.000]),
            high=np.array([1.0, 1.000]),
            shape=(2,),
            dtype=np.float32
        )


        # Assert that the render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def set_sock(self, sock: ACSocket):
        self._sock = sock

    def _update_obs(self):
        """
        Get the current observation from the game over socket.
        """
        # Send a request to the game
        self._sock.update()
        data = self._sock.data

        try:
            # Convert the byte data to a string
            data_str = data.decode('utf-8')

            # Split the string by commas and map values to a dictionary
            data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
        except:
            # If the data is invalid, throw error and return empty dict
            print(colorize("Error parsing data, returning empty dict!", "red"))
            return {}

        # throttle = float(data_dict['throttle'])
        # brake = float(data_dict['brake'])
        # steer = float(data_dict['steer'])
        # lap_time = float(data_dict['lap_time'])

        # print(data_dict)
        track_progress = float(data_dict['track_progress'])
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc_x = float(data_dict['world_loc[0]'])
        world_loc_y = float(data_dict['world_loc[1]'])
        world_loc_z = float(data_dict['world_loc[2]'])
        lap_count = float(data_dict['lap_count'])
        lap_time = int(data_dict['lap_time'])
        previous_track_progress = self._observations[0] if self._observations is not None else 0.000


        # Lap stays invalid as soon as it has been invalid once
        lap_invalid = self._invalid_flag
        if data_dict['lap_invalid'] == 'True':
            lap_invalid = 1.0
        self._invalid_flag = lap_invalid

        # Update the observations
        self._observations = np.array(
            [track_progress, speed_kmh, world_loc_x, world_loc_y, world_loc_z, lap_invalid, lap_count, lap_time, previous_track_progress], dtype=np.float32)
        return self._observations

    def _get_info(self):
        """
        Extra information returned by step and reset functions.
        """
        return {}

    def _get_reward(self):
        """
        TODO: IMPLEMENT REWARD FUNCTION
        """
        return

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initiate a new episode.
        :param seed: The seed for the environment's random number generator
        :param options: The options for the environment
        :return: The initial observation and info
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset the controller
        self.controller.reset_car()

        # Get the initial observations from the game
        self._invalid_flag = 0.0
        observation = self._update_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray, ignore_done: bool = False):
        """
        Perform an action in the environment and get the results.
        :param action: The action to perform
        :return: The observation, reward, terminated, truncated, info
        """
        # Perform the action in the game
        # print("action", action)
        self.controller.perform(action[0], action[1])

        # Get the new observations
        observation = self._update_obs()

        # Progress goal (100% of the track, with 0.99 to account for errors)
        progress_goal = 0.99

        lap_invalid = observation[5]
        lap_count = observation[6]
        track_progress = observation[0]
        if ignore_done:
            terminated = False
        else:
            terminated = lap_count > 1.0 or track_progress >= progress_goal

        # Truncated gets updated based on timesteps by TimeLimit wrapper
        truncated = False

        # Get the reward and info
        reward = None
        info = None
        if not ignore_done:
            reward = self._get_reward()
            info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment; a PyGame renderer is not needed for AC.
        """
        print(colorize("Rendering not supported for this environment!", "red"))

    def close(self):
        """
        Close the environment and the socket connection.
        """
        self._sock.end_training()
        self._sock.on_close()