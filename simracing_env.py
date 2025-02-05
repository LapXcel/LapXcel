#!/usr/bin/env python
"""
simracing_env.py

A custom OpenAI Gym environment for sim racing telemetry.
This environment is designed for reinforcement learning applications where
the agent controls steering, acceleration, and braking. The reward function
incentivizes optimizing lap time, efficient cornering, fuel usage, smooth inputs,
and penalizes wall contacts, tire degradation, and track limit violations.
"""

import gym
from gym import spaces
import numpy as np
import json
import socket
import time


class SimRacingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, telemetry_host='localhost', telemetry_port=5000):
        super(SimRacingEnv, self).__init__()
        
        # Define action space: originally continuous for [steering, acceleration, braking]
        # (We will discretize these actions in our agent.)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Define observation space:
        # Basic telemetry observation: [speed, acceleration, braking, steering, lap_progress]
        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Telemetry connection parameters
        self.telemetry_host = telemetry_host
        self.telemetry_port = telemetry_port
        self.telemetry_socket = None
        self.telemetry_buffer = ""

        # Internal variables for tracking progress and time
        self.current_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.last_progress = 0.0
        self.start_time = None
        self.done = False

        # Store the latest full telemetry dict for use in reward computation
        self.last_telemetry = {}

    def _connect_to_telemetry(self):
        try:
            self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.telemetry_socket.connect((self.telemetry_host, self.telemetry_port))
            self.telemetry_socket.settimeout(0.5)
            print("Connected to telemetry data source.")
        except Exception as e:
            print(f"Error connecting to telemetry: {e}")
            self.telemetry_socket = None

    def _disconnect_telemetry(self):
        if self.telemetry_socket:
            self.telemetry_socket.close()
            self.telemetry_socket = None

    def _read_telemetry(self):
        if not self.telemetry_socket:
            self._connect_to_telemetry()
        try:
            data = self.telemetry_socket.recv(1024).decode('utf-8')
            if data:
                self.telemetry_buffer += data
                if "\n" in self.telemetry_buffer:
                    line, self.telemetry_buffer = self.telemetry_buffer.split("\n", 1)
                    try:
                        telemetry = json.loads(line)
                        self.last_telemetry = telemetry
                        return telemetry
                    except json.JSONDecodeError:
                        return None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def _telemetry_to_observation(self, telemetry):
        try:
            observation = np.array([
                telemetry.get("speed", 0.0),
                telemetry.get("acceleration", 0.0),
                telemetry.get("braking", 0.0),
                telemetry.get("steering", 0.0),
                telemetry.get("lap_progress", 0.0)
            ], dtype=np.float32)
        except Exception as e:
            print(f"Error converting telemetry to observation: {e}")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation

    def reset(self):
        self.done = False
        self.last_progress = 0.0
        self._disconnect_telemetry()
        self._connect_to_telemetry()
        self.start_time = time.time()
        observation = self.get_initial_observation()
        self.current_observation = observation
        return observation

    def get_initial_observation(self):
        telemetry = self._read_telemetry()
        if telemetry is not None:
            return self._telemetry_to_observation(telemetry)
        else:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def send_action_to_game(self, action):
        # In a full implementation, send the action to the simulation/game.
        # For demonstration, we simply print the action.
        print(f"Action sent to game: {action}")

    def compute_reward(self, observation, action):
        reward = 0.0
        # --- Reward for Lap Time ---
        current_progress = observation[4]
        if current_progress >= 1.0:
            lap_time = time.time() - self.start_time
            lap_time_reward = 1000.0 / lap_time if lap_time > 0 else 0
            reward += lap_time_reward
            print(f"Lap completed in {lap_time:.2f}s, lap time reward: {lap_time_reward:.2f}")

        # --- Reward for Efficient Cornering ---
        corner_speed = self.last_telemetry.get("corner_speed", None)
        optimal_corner_speed = self.last_telemetry.get("optimal_corner_speed", None)
        if corner_speed is not None and optimal_corner_speed is not None:
            diff = abs(optimal_corner_speed - corner_speed)
            corner_reward = max(0, 50 - diff)
            reward += corner_reward

        # --- Reward for Hitting Apex ---
        if self.last_telemetry.get("apex_hit", False):
            reward += 50

        # --- Reward for Fuel Consumption Efficiency ---
        fuel_used = self.last_telemetry.get("fuel_used", None)
        if fuel_used is not None:
            fuel_efficiency_reward = max(0, 100 - fuel_used)
            reward += fuel_efficiency_reward

        # --- Reward for Throttle & Brake Efficiency ---
        smoothness_penalty = 0.1 * (abs(action[1]) + abs(action[2]))
        reward -= smoothness_penalty

        # --- Reward for DRS & ERS Utilization ---
        drs_usage = self.last_telemetry.get("drs_usage", 0.0)
        ers_usage = self.last_telemetry.get("ers_usage", 0.0)
        drs_reward = max(0, 20 - abs(drs_usage - 0.5) * 40)
        ers_reward = max(0, 20 - abs(ers_usage - 0.5) * 40)
        reward += drs_reward + ers_reward

        # --- Penalize Wall Contact, Tire Degradation, and Track Limit Violations ---
        if self.last_telemetry.get("wall_contact", False):
            reward -= 100
        tire_degradation = self.last_telemetry.get("tire_degradation", 0.0)
        reward -= tire_degradation * 0.5
        if self.last_telemetry.get("track_limit_violation", False):
            reward -= 150

        return reward

    def check_done(self, observation):
        if observation[4] >= 1.0:
            return True
        return False

    def step(self, action):
        self.send_action_to_game(action)
        time.sleep(0.05)
        telemetry = self._read_telemetry()
        if telemetry is not None:
            observation = self._telemetry_to_observation(telemetry)
        else:
            observation = self.current_observation
        reward = self.compute_reward(observation, action)
        self.done = self.check_done(observation)
        self.current_observation = observation
        info = {}
        return observation, reward, self.done, info

    def render(self, mode="human", close=False):
        print(f"Current observation: {self.current_observation}")

    def close(self):
        self._disconnect_telemetry()


if __name__ == "__main__":
    env = SimRacingEnv()
    obs = env.reset()
    print("Initial observation:", obs)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished.")
            break
    env.close()
