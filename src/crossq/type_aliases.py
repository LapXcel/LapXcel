from typing import NamedTuple, Any

import flax  # Importing the Flax library for neural network training
import numpy as np  # Importing NumPy for numerical operations
from flax.training.train_state import TrainState  # Importing TrainState class from Flax for managing training state

# Define a custom training state for the Actor component of the reinforcement learning model
class ActorTrainState(TrainState):
    # FrozenDict to hold batch statistics, which are immutable and can be used for statistics tracking
    batch_stats: flax.core.FrozenDict

# Define a custom training state for the Reinforcement Learning model
class RLTrainState(TrainState):  # type: ignore[misc]
    # FrozenDict to hold the target parameters of the model, which are used for stable learning
    target_params: flax.core.FrozenDict  # type: ignore[misc]
    # FrozenDict to hold batch statistics for the current training state
    batch_stats: flax.core.FrozenDict
    # FrozenDict to hold batch statistics for the target network, used in double Q-learning
    target_batch_stats: flax.core.FrozenDict

# Define a named tuple to store samples from the replay buffer
class ReplayBufferSamplesNp(NamedTuple):
    # Array of observations (states) taken from the environment
    observations: np.ndarray
    # Array of actions taken by the agent
    actions: np.ndarray
    # Array of next observations (states) after taking the actions
    next_observations: np.ndarray
    # Array indicating whether each episode has ended (1 for done, 0 for not done)
    dones: np.ndarray
    # Array of rewards received after taking the actions
    rewards: np.ndarray
