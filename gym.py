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
        
        # Define action space: steering, acceleration, and braking (continuous values)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Define observation space:
        # Basic telemetry observation: [speed, acceleration, braking, steering, lap_progress]
        # Additional telemetry will be used for reward computation but is not part of the observation vector.
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
        """Establish a TCP connection to the telemetry data stream."""
        try:
            self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.telemetry_socket.connect((self.telemetry_host, self.telemetry_port))
            self.telemetry_socket.settimeout(0.5)  # Non-blocking with timeout
            print("Connected to telemetry data source.")
        except Exception as e:
            print(f"Error connecting to telemetry: {e}")
            self.telemetry_socket = None

    def _disconnect_telemetry(self):
        """Close the telemetry socket if open."""
        if self.telemetry_socket:
            self.telemetry_socket.close()
            self.telemetry_socket = None

    def _read_telemetry(self):
        """
        Read data from the telemetry socket.
        Expected to receive JSON strings ending with a newline.
        Returns a dictionary of telemetry data.
        """
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
                        self.last_telemetry = telemetry  # store for reward computation
                        return telemetry
                    except json.JSONDecodeError:
                        return None
        except socket.timeout:
            return None
        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def _telemetry_to_observation(self, telemetry):
        """
        Convert raw telemetry dictionary to an observation numpy array.
        Expected telemetry keys (for observation): 'speed', 'acceleration', 'braking', 'steering', 'lap_progress'
        """
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
        """
        Reset the environment to an initial state and return an initial observation.
        """
        self.done = False
        self.last_progress = 0.0

        # Reset telemetry connection
        self._disconnect_telemetry()
        self._connect_to_telemetry()

        # Optionally, send a reset command to the simulation/game here.

        # Initialize lap timer
        self.start_time = time.time()

        # Get an initial observation
        observation = self.get_initial_observation()
        self.current_observation = observation
        return observation

    def get_initial_observation(self):
        """
        Retrieve an initial observation from the game telemetry.
        If no telemetry is available, return a default observation.
        """
        telemetry = self._read_telemetry()
        if telemetry is not None:
            return self._telemetry_to_observation(telemetry)
        else:
            # Return zeros with lap_progress = 0.0
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def send_action_to_game(self, action):
        """
        Send the chosen action to the simulation/game.
        This should interface with your game (e.g., via sockets, shared memory, or API calls).
        For now, this is a placeholder that prints the action.
        """
        # TODO: Replace with actual code to interface with the game.
        print(f"Action sent to game: {action}")

    def compute_reward(self, observation, action):
        """
        Compute the reward based on multiple factors:
        
         - Lap Time: Shorter lap times yield higher rewards.
         - Efficient Cornering: Reward maintaining optimal speed through corners.
         - Hitting Apex: Reward if the car’s tires hit the curb at the correct moment during a corner.
         - Fuel Consumption: Reward efficient fuel usage.
         - Throttle & Brake Efficiency: Reward smooth inputs.
         - DRS & ERS Utilization: Reward optimal usage of these systems.
         
        Penalize:
         - Wall Contact: Penalize any contact with barriers.
         - Tire Degradation: Penalize excessive tire wear.
         - Track Limit Violations: Heavily penalize if all four wheels leave the track.
        
        Telemetry data is assumed to include keys for these metrics. Adjust as needed.
        """
        reward = 0.0

        # ----- Reward for Lap Time -----
        # If the lap is completed (lap_progress >= 1.0), calculate lap time reward.
        current_progress = observation[4]
        if current_progress >= 1.0:
            lap_time = time.time() - self.start_time
            # The faster the lap, the higher the reward (using an inverse relation).
            lap_time_reward = 1000.0 / lap_time if lap_time > 0 else 0
            reward += lap_time_reward
            print(f"Lap completed in {lap_time:.2f}s, lap time reward: {lap_time_reward:.2f}")

        # ----- Reward for Efficient Cornering -----
        # Assume telemetry provides 'corner_speed' and 'optimal_corner_speed'
        corner_speed = self.last_telemetry.get("corner_speed", None)
        optimal_corner_speed = self.last_telemetry.get("optimal_corner_speed", None)
        if corner_speed is not None and optimal_corner_speed is not None:
            # Reward is higher if the actual corner speed is near the optimal value.
            diff = abs(optimal_corner_speed - corner_speed)
            corner_reward = max(0, 50 - diff)  # arbitrary scaling
            reward += corner_reward

        # ----- Reward for Hitting Apex -----
        # Assume telemetry provides a flag 'apex_hit' (True/False)
        if self.last_telemetry.get("apex_hit", False):
            reward += 50  # bonus for hitting the apex

        # ----- Reward for Fuel Consumption Efficiency -----
        # Assume telemetry provides 'fuel_used' for the lap.
        fuel_used = self.last_telemetry.get("fuel_used", None)
        if fuel_used is not None:
            # Less fuel used yields higher reward.
            fuel_efficiency_reward = max(0, 100 - fuel_used)  # arbitrary scaling
            reward += fuel_efficiency_reward

        # ----- Reward for Throttle & Brake Efficiency -----
        # Encourage smooth inputs: assume 'throttle' and 'brake' values and their rates of change.
        throttle = self.last_telemetry.get("throttle", 0.0)
        brake = self.last_telemetry.get("brake", 0.0)
        # For smoothness, you might compare the current value with a moving average or previous value.
        # Here, we use a placeholder penalty for abrupt inputs.
        smoothness_penalty = 0.1 * (abs(action[1]) + abs(action[2]))  # action[1]=acceleration, action[2]=brake
        reward -= smoothness_penalty

        # ----- Reward for DRS & ERS Utilization -----
        # Assume telemetry provides flags/values 'drs_usage' and 'ers_usage'
        drs_usage = self.last_telemetry.get("drs_usage", 0.0)
        ers_usage = self.last_telemetry.get("ers_usage", 0.0)
        # Optimal usage could be around a target value (say 0.5); reward proximity to that target.
        drs_reward = max(0, 20 - abs(drs_usage - 0.5) * 40)  # arbitrary scaling
        ers_reward = max(0, 20 - abs(ers_usage - 0.5) * 40)
        reward += drs_reward + ers_reward

        # ----- Penalizations -----
        # Wall Contact: assume telemetry 'wall_contact' is True if contact occurs.
        if self.last_telemetry.get("wall_contact", False):
            reward -= 100  # heavy penalty

        # Tire Degradation: assume telemetry provides a value 'tire_degradation' (0 to 100).
        tire_degradation = self.last_telemetry.get("tire_degradation", 0.0)
        reward -= tire_degradation * 0.5  # penalty proportional to degradation

        # Track Limit Violations: assume telemetry provides 'track_limit_violation'
        # If all four wheels off track, apply heavy penalty.
        if self.last_telemetry.get("track_limit_violation", False):
            reward -= 150

        return reward

    def check_done(self, observation):
        """
        Determine if the current episode is finished.
        In this example, the episode is done when:
         - The lap is complete (lap_progress >= 1.0).
         - (Additional conditions such as critical failure states could be added.)
        """
        if observation[4] >= 1.0:
            return True
        return False

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # Send the chosen action to the game.
        self.send_action_to_game(action)

        # Wait briefly for the game to update its state (adjust sleep time as needed).
        time.sleep(0.05)

        # Retrieve the latest telemetry observation.
        telemetry = self._read_telemetry()
        if telemetry is not None:
            observation = self._telemetry_to_observation(telemetry)
        else:
            observation = self.current_observation

        # Compute reward based on telemetry and action.
        reward = self.compute_reward(observation, action)

        # Check if the episode is done (lap complete or other conditions).
        self.done = self.check_done(observation)

        # Update the current observation.
        self.current_observation = observation

        # Optionally include additional info.
        info = {}

        return observation, reward, self.done, info

    def render(self, mode="human", close=False):
        """
        Render the environment.
        This method can be extended to visualize telemetry, track position, etc.
        """
        print(f"Current observation: {self.current_observation}")

    def close(self):
        """Clean up resources when the environment is closed."""
        self._disconnect_telemetry()

# Simple test run of the environment.
env = SimRacingEnv()
obs = env.reset()
print("Initial observation:", obs)

for _ in range(50):
    # Sample a random action.
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Obs: {obs}, Reward: {reward}, Done: {done}")
    if done:
        print("Episode finished.")
        break

env.close()