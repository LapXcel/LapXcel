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
    This class defines the environment in which the agent (car) operates.
    """

    metadata = {"render_modes": [], "render_fps": 0}  # Metadata about the environment
    _observations = None  # Placeholder for observations
    _invalid_flag = 0.0  # Flag to track if the lap is invalid
    _sock = None  # Socket for communication with the game

    def __init__(self, render_mode: Optional[str] = None, max_speed=200.0):
        self.steps_taken = 0  # Counter for steps taken in the environment
        # Define the observation space with limits for each observation
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2000.0, -2000.0, -2000.0, -max_speed, -max_speed]),
            high=np.array([max_speed, 2000.0, 2000.0, 2000.0, max_speed, max_speed]),
            shape=(6,),  # Six observations
            dtype=np.float32,
        )
        # Define the action space for the agent (car)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.000]),  # Brake/throttle and steering limits
            high=np.array([1.0, 1.000]),
            shape=(2,),
            dtype=np.float32
        )

        self.controller = ACController()  # Controller to manage car actions

        # Initialize reward-related variables
        self.max_speed = max_speed  # Maximum speed of the car
        self.track_progress = 0.000  # Progress on the track
        self.progress_goal = 0.99  # Target progress for completing the track
        self.lap_count = 0  # Counter for laps completed
        self.lap_time = 0  # Time taken for the current lap
        self.current_progress = 1  # Current progress milestone
        self.lap_invalid = False  # Flag to check if the lap is invalid

        # Observations include speed, world location, and velocities
        self.observation_space = spaces.Box(
            low=np.array([0.0, -2000.0, -2000.0, -2000.0, -max_speed, -max_speed]),
            high=np.array([max_speed, 2000.0, 2000.0, 2000.0, max_speed, max_speed]),
            shape=(6,),
            dtype=np.float32,
        )

        # Action space is defined as continuous for brake/throttle and steering
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.000]),
            high=np.array([1.0, 1.000]),
            shape=(2,),
            dtype=np.float32
        )

        # Ensure the specified render mode is valid
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def set_sock(self, sock: ACSocket) -> None:
        """
        Set the socket for the connection to the game.
        This allows communication for receiving observations and sending actions.
        """
        self._sock = sock

    def _update_obs(self) -> np.array:
        """
        Get the current observation from the game over the socket.
        This method retrieves the latest state of the car and track.
        """
        # Send a request to the game to get the current state
        self._sock.update()
        data = self._sock.data  # Receive data from the socket

        try:
            # Convert the byte data to a string
            data_str = data.decode('utf-8')

            # Split the string by commas and map values to a dictionary
            data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
        except:
            # If there is an error parsing the data, print an error message
            print(colorize("Error parsing data, returning empty dict!", "red"))
            return {}

        # Update variables for the reward function based on the parsed data
        self.track_progress = float(data_dict['track_progress'])
        self.lap_count = float(data_dict['lap_count'])
        self.lap_time = int(data_dict['lap_time'])

        # Check if the lap is invalid based on the received data
        self.lap_invalid = self._invalid_flag
        if data_dict['lap_invalid'] == 'True':
            self.lap_invalid = 1.0
        self._invalid_flag = self.lap_invalid

        # Retrieve observations from the data dictionary
        speed_kmh = float(data_dict['speed_kmh'])
        world_loc_x = float(data_dict['world_loc[0]'])
        world_loc_y = float(data_dict['world_loc[1]'])
        world_loc_z = float(data_dict['world_loc[2]'])
        velocity_x = float(data_dict['velocity[0]'])
        velocity_z = float(data_dict['velocity[1]'])

        # Update the observations array with the latest values
        self._observations = np.array(
            [speed_kmh, world_loc_x, world_loc_y, world_loc_z, velocity_x, velocity_z], dtype=np.float32)
        return self._observations

    def _get_info(self) -> dict:
        """
        Extra information returned by step and reset functions.
        This can be used for debugging or logging purposes.
        """
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to an initial state.
        This function is called at the beginning of each episode.
        """
        super().reset(seed=seed)  # Call the parent class's reset method
        self.controller.reset_car()  # Reset the car's state
        self._invalid_flag = 0.0  # Reset the invalid lap flag
        observation = self._update_obs()  # Get the initial observation
        info = self._get_info()  # Get additional info
        return observation, info  # Return the initial observation and info

    def step(self, action: np.ndarray, ignore_done: bool = False):
        """
        Apply an action to the environment and return the new state.
        This function is called at each time step.
        """
        # Apply the action in the game using the controller
        self.controller.perform(action[0], action[1])
        
        observation = self._update_obs()  # Get the new observation after action

        # Check termination conditions for the episode
        terminated = (self.track_progress >= self.progress_goal and self.lap_count >= 2
                    or self.lap_invalid)

        truncated = False  # Not used; can be implemented if needed
        reward = self._get_reward(terminated)  # Calculate the reward based on the current state
        info = self._get_info()  # Retrieve additional info

        return observation, reward, terminated, truncated, info  # Return the new state, reward, and info

    def _get_reward(self, terminated: bool) -> float:
        """
        Calculate the reward based on the current state of the environment.
        The reward structure encourages progress and penalizes invalid laps.
        """
        step_penalty = 0.01  # Penalty for each step taken
        progress_reward = 0.01  # Reward for progressing on the track
        finishing_reward = 1.0  # Reward for finishing the lap

        # Determine the reward based on termination conditions
        if terminated:
            if self.lap_invalid:
                return -1000.0  # Large penalty for invalid lap
            else:
                return finishing_reward  # Reward for finishing the lap
        elif self.track_progress * 100 >= self.current_progress:
            temp = self.current_progress
            self.current_progress += 1  # Update progress milestone
            return -step_penalty + progress_reward * temp  # Reward for progress
        else:
            return -step_penalty  # Penalty for taking a step without progress

    def render(self) -> None:
        """
        Render the environment; no rendering support is provided for this environment.
        This method is a placeholder and can be modified if rendering is needed.
        """
        print(colorize("Rendering not supported for this environment!", "red"))

    def close(self) -> None:
        """
        Close the environment and the socket connection.
        This method is called when the environment is no longer needed.
        """
        self._sock.end_training()  # Signal the end of training to the socket
        self._sock.on_close()  # Close the socket connection
