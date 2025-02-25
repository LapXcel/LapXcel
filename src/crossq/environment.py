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

        # Initialize reward variables
        self.max_speed = max_speed
        self.track_progress = 0.000
        self.progress_goal = 0.99
        self.lap_count = 0
        self.lap_time = 0
        self.lap_invalid = False

        # Observations is a Box with the following data:
        # - "speed_kmh": The speed of the car in km/h [0.0, max_speed]
        # - "world_loc_x": The world's x location of the car [-2000.0, 2000.0]
        # - "world_loc_y": The world's y location of the car [-2000.0, 2000.0]
        # - "world_loc_z": The world's z location of the car [-2000.0, 2000.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2000.0, -2000.0, -2000.0]),
            high=np.array([max_speed, 2000.0, 2000.0, 2000.0]),
            shape=(4,),
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

    def set_sock(self, sock: ACSocket) -> None:
        """
        Set the socket for the socket connection.
        """
        self._sock = sock

    def _update_obs(self) -> np.array:
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

        # Update variables for reward function
        self.track_progress = float(data_dict['track_progress'])
        self.lap_count = float(data_dict['lap_count'])
        self.lap_time = int(data_dict['lap_time'])

        # Lap stays invalid as soon as it has been invalid once
        self.lap_invalid = self._invalid_flag
        if data_dict['lap_invalid'] == 'True':
            self.lap_invalid = 1.0
        self._invalid_flag = self.lap_invalid

        # Get observation space
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc_x = float(data_dict['world_loc[0]'])
        world_loc_y = float(data_dict['world_loc[1]'])
        world_loc_z = float(data_dict['world_loc[2]'])

        # Update the observations
        self._observations = np.array(
            [speed_kmh, world_loc_x, world_loc_y, world_loc_z], dtype=np.float32)
        return self._observations

    def _get_info(self) -> dict:
        """
        Extra information returned by step and reset functions.
        """
        return {}

    def _get_reward(self, terminated: bool) -> int:
        """
        Reward function for the agent. It will give a reward of
        120001 (2 minutes) - lap_time if the lap has been completed.
        """
        if terminated:
            return 120001 - self.lap_time if not self.lap_invalid else 0

        return 0

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

        if ignore_done:
            terminated = False
        else:
            terminated = (self.lap_count > 1.0 
                            or self.track_progress >= self.progress_goal 
                            or self.lap_time >= 120000
                            or self.lap_invalid)

        # Truncated gets updated based on timesteps by TimeLimit wrapper
        truncated = False

        # Get the reward and info
        reward = None
        info = None
        if not ignore_done:
            reward = self._get_reward(terminated)
            info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Render the environment; a PyGame renderer is not needed for AC.
        """
        print(colorize("Rendering not supported for this environment!", "red"))

    def close(self) -> None:
        """
        Close the environment and the socket connection.
        """
        self._sock.end_training()
        self._sock.on_close()