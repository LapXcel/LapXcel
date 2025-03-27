from typing import Any, Dict, List, Optional, Tuple, Type, Union

import jax  # JAX for high-performance numerical computing
import numpy as np  # NumPy for numerical operations
from gymnasium import spaces  # Gymnasium library for defining environments
from stable_baselines3 import HerReplayBuffer  # HER (Hindsight Experience Replay) buffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer  # Replay buffers
from stable_baselines3.common.noise import ActionNoise  # Action noise for exploration
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm  # Base class for off-policy algorithms
from stable_baselines3.common.policies import BasePolicy  # Base class for policies
from stable_baselines3.common.type_aliases import GymEnv, Schedule  # Type aliases for environment and schedule

class OffPolicyAlgorithmJax(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],  # Policy type to use
        env: Union[GymEnv, str],  # Environment to train on
        learning_rate: Union[float, Schedule],  # Learning rate or learning rate schedule
        qf_learning_rate: Optional[float] = None,  # Learning rate for the Q-function
        buffer_size: int = 1_000_000,  # Size of the replay buffer
        learning_starts: int = 100,  # Steps before learning starts
        batch_size: int = 256,  # Number of samples per batch
        tau: float = 0.005,  # Soft update coefficient for target networks
        gamma: float = 0.99,  # Discount factor
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),  # Frequency of training
        gradient_steps: int = 1,  # Number of gradient steps per training call
        action_noise: Optional[ActionNoise] = None,  # Noise added to actions for exploration
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,  # Custom replay buffer class
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,  # Additional arguments for replay buffer
        optimize_memory_usage: bool = False,  # Flag to optimize memory usage
        policy_kwargs: Optional[Dict[str, Any]] = None,  # Extra arguments for the policy
        tensorboard_log: Optional[str] = None,  # Directory for TensorBoard logging
        verbose: int = 0,  # Verbosity level
        device: str = "auto",  # Device to run the algorithm on (CPU or GPU)
        support_multi_env: bool = False,  # Flag to support multiple environments
        monitor_wrapper: bool = True,  # Flag to use monitoring wrapper
        seed: Optional[int] = None,  # Random seed for reproducibility
        use_sde: bool = False,  # Flag to use Stochastic Differential Equations
        sde_sample_freq: int = -1,  # Frequency of SDE sampling
        use_sde_at_warmup: bool = False,  # Flag to use SDE during warmup
        sde_support: bool = True,  # Flag to enable SDE support
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,  # Supported action spaces
        stats_window_size: int = 100,  # Size of the window for statistics
    ):
        # Initialize the parent class with the provided parameters
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=support_multi_env,
            stats_window_size=stats_window_size,
        )
        
        # Initialize the random key for JAX (for reproducibility)
        self.key = jax.random.PRNGKey(0)
        # Set the Q-function learning rate; not allowed to be a schedule
        self.qf_learning_rate = qf_learning_rate

    def _get_torch_save_params(self):
        # Return parameters that should be saved for the Torch model
        return [], []

    def _excluded_save_params(self) -> List[str]:
        # Get parameters to be excluded from saving
        excluded = super()._excluded_save_params()
        excluded.remove("policy")  # Always include the policy in saving
        return excluded

    def set_random_seed(self, seed: Optional[int]) -> None:  # type: ignore[override]
        # Set the random seed for reproducibility
        super().set_random_seed(seed)
        if seed is None:
            # If no seed is provided, sample a random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)  # Update the PRNG key with the new seed

    def _setup_model(self) -> None:
        # Set up the model and replay buffer
        if self.replay_buffer_class is None:  # Check if a custom replay buffer class is provided
            # Choose replay buffer class based on observation space type
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer  # Use DictReplayBuffer for dict observation spaces
            else:
                self.replay_buffer_class = ReplayBuffer  # Use standard ReplayBuffer

        self._setup_lr_schedule()  # Set up the learning rate schedule
        # Default the Q-function learning rate to the policy learning rate if not specified
        self.qf_learning_rate = self.qf_learning_rate or self.lr_schedule(1)
        self.set_random_seed(self.seed)  # Set the random seed for the model

        # Make a local copy of replay buffer kwargs; avoid pickling the environment
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        # If using HER, ensure the environment is passed to the replay buffer
        if issubclass(self.replay_buffer_class, HerReplayBuffer):  # Check if HER is used
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env  # Add environment to kwargs

        # Initialize the replay buffer with the specified parameters
        self.replay_buffer = self.replay_buffer_class(  # Create an instance of the replay buffer
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device="cpu",  # Force CPU device for easy conversion between Torch and NumPy
            n_envs=self.n_envs,  # Number of environments
            optimize_memory_usage=self.optimize_memory_usage,  # Flag for memory optimization
            **replay_buffer_kwargs,  # Unpack additional kwargs for the replay buffer
        )
        # Convert training frequency parameter to a TrainFreq object for internal use
        self._convert_train_freq()
