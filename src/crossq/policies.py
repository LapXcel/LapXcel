from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type, Tuple, no_type_check
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation
from crossq.distributions import TanhTransformedDistribution
from crossq.type_aliases import RLTrainState, ActorTrainState

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

from flax.linen.normalization import _compute_stats, _normalize, _canonicalize_axes
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact, merge_param  # pylint: disable=g-multiple-import
from jax import lax
from jax.nn import initializers

# Type aliases for better code readability
PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any  # Placeholder for data type
Axes = Union[int, Sequence[int]]

class BaseJaxPolicy(BasePolicy):
    """Base class for JAX-based policies, extending from the BasePolicy."""
    
    def __init__(self, *args, **kwargs):
        # Initialize the base policy with provided arguments
        super().__init__(*args, **kwargs)

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def sample_action(actor_state, observations, key, return_logprob=False):
        """Sample an action from the policy distribution."""
        # Apply the actor's function to get the distribution
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        observations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, observations)
        
        # Sample an action from the distribution
        action = dist.sample(seed=key)
        
        # Return the sampled action and optionally the log probability
        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @staticmethod
    @partial(jax.jit, static_argnames=["return_logprob"])
    def select_action(actor_state, observations, return_logprob=False):
        """Select the action with the highest probability (mode) from the policy distribution."""
        # Similar to sample_action, but returns the mode of the distribution
        if hasattr(actor_state, "batch_stats"):
            dist = actor_state.apply_fn({"params": actor_state.params, "batch_stats": actor_state.batch_stats},
                                        observations, train=False)
        else:
            dist = actor_state.apply_fn(actor_state.params, observations)
        
        action = dist.mode()

        # Return the selected action and optionally the log probability
        if not return_logprob:
            return action
        else:
            return action, dist.log_prob(action)

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Predict actions based on the given observations."""
        # Prepare the observations (handle dicts, etc.)
        observation, vectorized_env = self.prepare_obs(observation)

        # Get predicted actions from the policy
        actions = self._predict(observation, deterministic=deterministic)

        # Convert actions to the appropriate shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip and rescale actions if output is squashed
                actions = np.clip(actions, -1, 1)
                actions = self.unscale_action(actions)
            else:
                # Ensure actions are within the action space bounds
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Squeeze batch dimension if not using a vectorized environment
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        """Prepare observations for the model, handling different formats."""
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Flatten the dictionary of observations
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(observation[keys[0]], self.observation_space[keys[0]])

            # Concatenate observations from the dictionary into a single array
            observation = np.concatenate(
                [observation[key].reshape(-1, *self.observation_space[key].shape) for key in keys],
                axis=1,
            )

        elif is_image_space(self.observation_space):
            # Handle image observations by transposing if necessary
            observation = maybe_transpose(observation, self.observation_space)

        else:
            # Convert generic observations to numpy array
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        """Set the training mode for the policy (affects layers like dropout)."""
        # self.actor.set_training_mode(mode)  # Uncomment if needed
        # self.critic.set_training_mode(mode)  # Uncomment if needed
        self.training = mode

class BatchRenorm(Module):
    """BatchRenorm Module, implementing Batch Renormalization as described in the paper.
    
    Attributes:
        use_running_average: If True, use stored statistics instead of computing from the input.
        axis: The axis along which to normalize.
        momentum: Decay rate for moving averages of statistics.
        epsilon: Small float added to variance to avoid division by zero.
        dtype: Data type of the result.
        param_dtype: Data type for parameter initializers.
        use_bias: If True, add a bias term.
        use_scale: If True, scale by a factor.
        bias_init: Initializer for the bias term.
        scale_init: Initializer for the scale term.
        axis_name: Name used for combining statistics across devices.
        axis_index_groups: Groups of indices for independent normalization across devices.
        use_fast_variance: If true, use a faster but less stable variance calculation.
    """
    
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.999
    epsilon: float = 0.001
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalize the input tensor x."""
        # Determine if to use running averages or compute new statistics
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        # Initialize running averages for mean and variance
        ra_mean = self.variable(
            'batch_stats',
            'mean',
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            'batch_stats', 'var', lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        # Initialize other batch renormalization variables
        r_max = self.variable(
            'batch_stats',
            'r_max',
            lambda s: s,
            3,
        )
        d_max = self.variable(
            'batch_stats',
            'd_max',
            lambda s: s,
            5,
        )
        steps = self.variable(
            'batch_stats',
            'steps',
            lambda s: s,
            0,
        )

        if use_running_average:
            # Use stored statistics if running average is enabled
            mean, var = ra_mean.value, ra_var.value
            custom_mean = mean
            custom_var = var
        else:
            # Compute new statistics from the input
            mean, var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            custom_mean = mean
            custom_var = var
            
            if not self.is_initializing():
                # Implement Batch Renormalization adjustments
                r = 1
                d = 0
                std = jnp.sqrt(var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                d = jax.lax.stop_gradient((mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)
                tmp_var = var / (r**2)
                tmp_mean = mean - d * jnp.sqrt(custom_var) / r

                # Warm up batch renorm for 100,000 steps
                warmed_up = jnp.greater_equal(steps.value, 100_000).astype(jnp.float32)
                custom_var = warmed_up * tmp_var + (1. - warmed_up) * custom_var
                custom_mean = warmed_up * tmp_mean + (1. - warmed_up) * custom_mean

                # Update running averages
                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var
                steps.value += 1

        # Normalize the input using the computed or running statistics
        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )

class Critic(nn.Module):
    """Critic neural network module for estimating state-action values."""
    
    net_arch: Sequence[int]  # Architecture of the neural network
    activation_fn: Type[nn.Module]  # Activation function to use
    batch_norm_momentum: float  # Momentum for batch normalization
    use_layer_norm: bool = False  # Flag to use layer normalization
    dropout_rate: Optional[float] = None  # Dropout rate for regularization
    use_batch_norm: bool = False  # Flag to use batch normalization
    bn_mode: str = "bn"  # Mode for batch normalization

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train) -> jnp.ndarray:
        """Forward pass through the Critic network."""
        # Select batch normalization class based on mode
        if 'bn' in self.bn_mode:
            BN = nn.BatchNorm
        elif 'brn' in self.bn_mode:
            BN = BatchRenorm
        else:
            raise NotImplementedError

        # Concatenate state and action inputs
        x = jnp.concatenate([x, action], -1)

        # Apply batch normalization if enabled
        if self.use_batch_norm:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Dummy pass to ensure consistent function signature
            x_dummy = BN(use_running_average=not train)(x)

        # Build the network architecture
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)  # Dense layer
            
            # Apply dropout if specified
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            
            # Apply layer normalization if specified
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)

            # Apply activation function
            x = self.activation_fn()(x)

            # Apply batch normalization again if needed
            if self.use_batch_norm:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)

        # Final output layer to produce state-action value (Q-value)
        x = nn.Dense(1)(x)
        return x

class VectorCritic(nn.Module):
    """Vectorized Critic that enables multiple Q-value estimators."""
    
    net_arch: Sequence[int]  # Network architecture
    activation_fn: Type[nn.Module]  # Activation function
    batch_norm_momentum: float  # Momentum for batch normalization
    use_batch_norm: bool = False  # Flag to use batch normalization
    batch_norm_mode: str = "bn"  # Mode for batch normalization
    use_layer_norm: bool = False  # Flag for layer normalization
    dropout_rate: Optional[float] = None  # Dropout rate
    n_critics: int = 2  # Number of critics to use

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = True):
        """Forward pass through the VectorCritic."""
        # Vectorized mapping of the Critic class
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "dropout": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        # Apply the vectorized critic to get Q-values
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, train)
        return q_values

class Actor(nn.Module):
    """Actor neural network module for generating actions."""
    
    net_arch: Sequence[int]  # Architecture of the neural network
    action_dim: int  # Dimension of the action space
    batch_norm_momentum: float  # Momentum for batch normalization
    log_std_min: float = -20  # Minimum log standard deviation
    log_std_max: float = 2  # Maximum log standard deviation
    use_batch_norm: bool = False  # Flag to use batch normalization
    bn_mode: str = "bn"  # Mode for batch normalization

    def get_std(self):
        """Get standard deviation for gSDE."""
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train) -> tfd.Distribution:
        """Forward pass through the Actor network."""
        # Select batch normalization class based on mode
        if 'brn_actor' in self.bn_mode:
            BN = BatchRenorm
        elif 'bn' in self.bn_mode or 'brn' in self.bn_mode:
            BN = nn.BatchNorm
        else:
            raise NotImplementedError

        # Apply batch normalization if enabled
        if self.use_batch_norm and not 'noactor' in self.bn_mode:
            x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
        else:
            # Dummy pass to ensure consistent function signature
            x_dummy = BN(use_running_average=not train)(x)

        # Build the network architecture
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)  # Dense layer
            x = nn.relu(x)  # Activation function
            
            # Apply batch normalization if enabled
            if self.use_batch_norm and not 'noactor' in self.bn_mode:
                x = BN(use_running_average=not train, momentum=self.batch_norm_momentum)(x)
            else:
                x_dummy = BN(use_running_average=not train)(x)

        # Output layers for mean and log standard deviation
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)  # Clip log_std to defined range
        
        # Create a Tanh-transformed distribution for action sampling
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist

class SACPolicy(BaseJaxPolicy):
    """Soft Actor-Critic (SAC) policy implementation."""
    
    action_space: spaces.Box  # Define the action space

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        activation_fn: Type[nn.Module],
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = False,
        batch_norm_momentum: float = 0.9,
        batch_norm_mode: str = "bn",
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        td3_mode: bool = False,
    ):
        # Initialize the base policy
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        # Store parameters for the SAC policy
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_mode = batch_norm_mode
        self.activation_fn = activation_fn
        
        # Set up network architecture
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        
        self.n_critics = n_critics
        self.use_sde = use_sde

        # Initialize random keys for JAX operations
        self.key = self.noise_key = jax.random.PRNGKey(0)

        # If TD3 mode is enabled, use deterministic prediction
        if td3_mode:
            self._predict = self._predict_deterministic

    def build(self, key: jax.random.KeyArray, lr_schedule: Schedule, qf_learning_rate: float) -> jax.random.KeyArray:
        """Build the SAC policy by initializing the actor and critic networks."""
        # Split keys for different components
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key, 2)  # Keep a key for the actor

        # Initialize noise for exploration
        self.reset_noise()

        # Create dummy observations for initialization
        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        # Initialize the Actor network
        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            bn_mode=self.batch_norm_mode,
        )
        # Hack to enable noise resetting for gSDE
        self.actor.reset_noise = self.reset_noise

        # Initialize actor parameters
        actor_init_variables = self.actor.init(
            {"params": actor_key, "batch_stats": bn_key},
            obs,
            train=False
        )
        # Create actor training state
        self.actor_state = ActorTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_init_variables["params"],
            batch_stats=actor_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # Initialize the VectorCritic network
        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            batch_norm_mode=self.batch_norm_mode,
            net_arch=self.net_arch_qf,
            activation_fn=self.activation_fn,
            n_critics=self.n_critics,
        )

        # Initialize critic parameters
        qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        target_qf_init_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        # Create critic training state
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_init_variables["params"],
            batch_stats=qf_init_variables["batch_stats"],
            target_params=target_qf_init_variables["params"],
            target_batch_stats=target_qf_init_variables["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        # JIT compile the apply functions for efficiency
        self.actor.apply = jax.jit(  # type: ignore[method-assign]
            self.actor.apply,
            static_argnames=("use_batch_norm", "batch_norm_momentum", "bn_mode")
        )
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm",
                             "use_batch_norm", "batch_norm_momentum", "bn_mode"),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new weights for the exploration matrix when using gSDE."""
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Forward pass to predict actions."""
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict actions based on observations."""
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        
        # Reset noise for exploration if not using gSDE
        if not self.use_sde:
            self.reset_noise()
        
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    def _predict_deterministic(self, observation: np.ndarray, **kwargs) -> np.ndarray:
        """Deterministic action prediction."""
        return BaseJaxPolicy.select_action(self.actor_state, observation)

    def predict_action_with_logprobs(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict actions along with their log probabilities."""
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation, True)
        
        # Reset noise for exploration if not using gSDE
        if not self.use_sde:
            self.reset_noise()
        
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key, True)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict Q-values for given observations and actions."""
        if not self.use_sde:
            self.reset_noise()

        def Q(params, batch_stats, o, a, dropout_key):
            """Internal function to compute Q-values."""
            return self.qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                o, a, 
                rngs={"dropout": dropout_key},
                train=False
            ) 
        
        return jax.jit(Q)(
            self.qf_state.params, 
            self.qf_state.batch_stats, 
            observation, 
            action,
            self.noise_key,
        )
