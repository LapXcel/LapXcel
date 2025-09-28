# Original author: Roma Sokolkov
# Edited by Antonin Raffin and Colby Todd
import warnings

import gymnasium
import numpy as np
from gymnasium import spaces

from config import INPUT_DIM, MIN_STEERING, MAX_STEERING, JERK_REWARD_WEIGHT, MAX_STEERING_DIFF
from envs.ac_sim import ACController
from gymnasium.spaces.utils import flatten_space, flatten

class ACVAEEnv(gymnasium.Env):
    """
    Gym interface for Assetto Corsa with support for using
    a VAE encoded observation instead of raw pixels if needed.

    :param frame_skip: (int) frame skip, also called action repeat
    :param vae: (VAEController object)
    :param const_throttle: (float) If set, the car only controls steering
    :param min_throttle: (float)
    :param max_throttle: (float)
    :param n_command_history: (int) number of previous commmands to keep
        it will be concatenated with the vae latent vector
    :param n_stack: (int) Number of frames to stack (used in teleop mode only)
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, frame_skip=2, vae=None, const_throttle=None,
                 min_throttle=-1.0, max_throttle=1.0, n_command_history=0,
                 n_stack=1, verbose=False):
        self.verbose = verbose
        self.vae = vae
        self.z_size = None
        if vae is not None:
            self.z_size = vae.z_size

        self.const_throttle = const_throttle
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.np_random = None

        # Save last n commands (throttle + steering)
        self.n_commands = 2
        self.command_history = np.zeros((1, self.n_commands * n_command_history))
        self.n_command_history = n_command_history
        # Custom frame-stack
        self.n_stack = n_stack
        self.stacked_obs = None

        # TCP port for communicating with simulation
        port = 9999

        # start simulation com
        self.viewer = ACController(port=port, verbose=self.verbose)

        if const_throttle is not None:
            # steering only
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING]),
                                           high=np.array([MAX_STEERING]),
                                           dtype=np.float32)
        else:
            # steering + throttle, action space must be symmetric
            self.action_space = spaces.Box(low=np.array([-MAX_STEERING, self.min_throttle]),
                                           high=np.array([MAX_STEERING, self.max_throttle]), dtype=np.float32)

        if vae is None:
            # Using pixels as input
            if n_command_history > 0:
                warnings.warn("n_command_history not supported for images"
                              "(it will not be concatenated with the input)")
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8),               # image_array
                spaces.Box(low=-900, high=900, shape=(1,), dtype=np.float32),              # steering_angle
                spaces.Box(low=-100, high=400, shape=(1,), dtype=np.float32),              # speed
                spaces.Box(low=-100, high=+100, shape=(3,), dtype=np.float32),           # velocity
                spaces.Box(low=-100, high=+100, shape=(3,), dtype=np.float32),           # acceleration
            ))
            self.observation_space = flatten_space(self.observation_space)
        else:
            # z latent vector from the VAE (encoded input image)
            self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                                high=np.finfo(np.float32).max,
                                                shape=(1, self.z_size + self.n_commands * n_command_history),
                                                dtype=np.float32)

        # Frame-stacking with teleoperation
        if n_stack > 1:
            obs_space = self.observation_space
            low = np.repeat(obs_space.low, self.n_stack, axis=-1)
            high = np.repeat(obs_space.high, self.n_stack, axis=-1)
            self.stacked_obs = np.zeros(low.shape, low.dtype)
            self.observation_space = spaces.Box(low=low, high=high, dtype=obs_space.dtype)
            # else:
            #     obs_space = self.observation_space  # This is spaces.Tuple with Boxes inside

            #     # Create lists to hold stacked low/high arrays for each Box in the Tuple
            #     low_list = []
            #     high_list = []

            #     for space in obs_space.spaces:
            #         # Repeat low and high along the last axis for stacking
            #         stacked_low = np.repeat(space.low, n_stack, axis=-1)
            #         stacked_high = np.repeat(space.high, n_stack, axis=-1)
            #         low_list.append(stacked_low)
            #         high_list.append(stacked_high)

            #     # Create new Tuple observation space with stacked Boxes
            #     stacked_spaces = tuple(
            #         spaces.Box(low=l, high=h, dtype=space.dtype) 
            #         for l, h, space in zip(low_list, high_list, obs_space.spaces)
            #     )
            #     self.observation_space = spaces.Tuple(stacked_spaces)

            #     # Initialize stacked_obs as a list of arrays for each component
            #     self.stacked_obs = [np.zeros_like(l, dtype=space.dtype) for l, space in zip(low_list, obs_space.spaces)]

        # Frame Skipping
        self.frame_skip = frame_skip
        # wait until loaded
        self.viewer.wait_until_loaded()
        if self.verbose:
            print("[ENV] Environment initialized.")

    def close_connection(self):
        return self.viewer.close_connection()

    def exit_scene(self):
        self.viewer.handler.send_exit_scene()

    def jerk_penalty(self):
        """
        Add a continuity penalty to limit jerk.
        :return: (float)
        """
        jerk_penalty = 0
        if self.n_command_history > 1:
            # Take only last command into account
            for i in range(1):
                steering = self.command_history[0, -2 * (i + 1)]
                prev_steering = self.command_history[0, -2 * (i + 2)]
                steering_diff = (prev_steering - steering) / (MAX_STEERING - MIN_STEERING)

                if abs(steering_diff) > MAX_STEERING_DIFF:
                    error = abs(steering_diff) - MAX_STEERING_DIFF
                    jerk_penalty += JERK_REWARD_WEIGHT * (error ** 2)
                else:
                    jerk_penalty += 0
        return jerk_penalty

    def postprocessing_step(self, action, observation, reward, done, truncated, info):
        """
        Update the reward (add jerk_penalty if needed), the command history
        and stack new observation (when using frame-stacking).

        :param action: ([float])
        :param observation: (np.ndarray)
        :param reward: (float)
        :param done: (bool)
        :param info: (dict)
        :return: (np.ndarray, float, bool, dict)
        """
        if self.verbose:
            print("[ENV] Postprocessing...")
        # Update command history
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((observation, self.command_history), axis=-1)

        jerk_penalty = self.jerk_penalty()
        # Cancel reward if the continuity constrain is violated
        if jerk_penalty > 0 and reward > 0:
            reward = 0
        reward -= jerk_penalty

        if self.n_stack > 1:
            observation = np.concatenate([elem.flatten() for elem in observation])
            observation = flatten(self.observation_space, observation)
            self.stacked_obs = np.roll(self.stacked_obs, shift=-observation.shape[-1], axis=-1)
            if done:
                self.stacked_obs[...] = 0
            self.stacked_obs[..., -observation.shape[-1]:] = observation
            return self.stacked_obs, reward, done, truncated, info

        return observation, reward, done, truncated, info

    def step(self, action):
        """
        :param action: (np.ndarray)
        :return: (np.ndarray, float, bool, dict)
        """
        if self.verbose:
            print("[ENV] Taking step...")
        # action[0] is the steering angle
        # action[1] is the throttle
        if action[1] > 0:
            action[1] = action[1] / self.max_throttle
        else:
            action[1] = action[1] / self.min_throttle
        
        # action[1] = math.tanh(*action[1]) # Smooth out the action to prefer the extremes

        if self.const_throttle is not None:
            action = np.concatenate([action, [self.const_throttle]])
        # else:
        #     # Convert from [-1, 1] to [0, 1]
        #     t = (action[1] + 1) / 2
        #     # Convert from [0, 1] to [min, max]
        #     action[1] = (1 - t) * self.min_throttle + self.max_throttle * t

        # Clip steering angle rate to enforce continuity
        # print(f"Action taken before {action[0]}")
        # action[0] = 0.15*action[0]
        # print(f"Action taken after expo {action[0]}")
        # if self.n_command_history > 0:
        #     prev_steering = self.command_history[0, -2]
        #     # print(f"Previous value {prev_steering}")
        #     max_diff = (MAX_STEERING_DIFF - 1e-5) * (MAX_STEERING - MIN_STEERING)
        #     diff = np.clip(action[0] - prev_steering, -max_diff, max_diff)
        #     # action[0] += prev_steering
        #     # action[0] = min(max(action[0], MIN_STEERING), MAX_STEERING)
        #     action[0] = diff + prev_steering
        # print(f"Action taken after {action[0]}")
        # Repeat action if using frame_skip
        for _ in range(self.frame_skip):
            if self.verbose:
                print("[ENV] Performing action...")
            self.viewer.take_action(action)
            observation, reward, done, truncated, info = self.observe()

            # if done:
            #     print("[ENV] Episode done.")
            #     observation, info = self.reset()
            #     break
        return self.postprocessing_step(action, observation, reward, done, truncated, info)

    def reset(self, seed = None, options = None):
        if self.verbose:
            print("[ENV] Resetting environment...")
        self.viewer.reset()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation, reward, done, truncated, info = self.observe()

        if not self.vae:
            observation = np.concatenate([elem.flatten() for elem in observation])
            print(observation)
            flatten(self.observation_space, observation)

        if self.n_command_history > 0:
            observation = np.concatenate((observation, self.command_history), axis=-1)

        if self.n_stack > 1:
            if self.vae:
                self.stacked_obs[...] = 0
                self.stacked_obs[..., -observation.shape[-1]:] = observation
            else:
                self.stacked_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return self.stacked_obs, info

        return observation, info

    def render(self, mode='human'):
        """
        :param mode: (str)
        """
        if mode == 'rgb_array':
            return self.viewer.handler.original_image
        return None

    def observe(self):
        """
        Encode the observation using VAE if needed.

        :return: (np.ndarray, float, bool, dict)
        """
        if self.verbose:
            print("[ENV] Getting observation...")
        observation, reward, done, truncated, info = self.viewer.observe()
        # Learn from Pixels
        if self.vae is None:
            return observation, reward, done, truncated, info

        # Encode the image
        observation = observation.astype(np.float32) / 255.0
        return self.vae.encode(observation)[0], reward, done, truncated, info

    def close(self):
        self.viewer.quit()

    def set_vae(self, vae):
        """
        :param vae: (VAEController object)
        """
        self.vae = vae