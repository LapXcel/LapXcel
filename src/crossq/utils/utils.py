import os  # Import the os module for interacting with the operating system

import jax  # Import JAX for high-performance numerical computing
import jax.numpy as jnp  # Import JAX's numpy module for array operations
import flax.linen as nn  # Import Flax's neural network library for building neural networks

def is_slurm_job():
    """Checks whether the script is run within a SLURM job scheduler environment."""
    # Check if any environment variable starts with 'SLURM'
    return bool(len({k: v for k, v in os.environ.items() if 'SLURM' in k}))


# Define a ReLU activation function as a Flax module
class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)  # Apply the ReLU activation function

# Define a ReLU6 activation function as a Flax module
class ReLU6(nn.Module):
    def __call__(self, x):
        return nn.relu6(x)  # Apply the ReLU6 activation function

# Define a Tanh activation function as a Flax module
class Tanh(nn.Module):
    def __call__(self, x):
        return nn.tanh(x)  # Apply the Tanh activation function

# Define a Sin activation function as a Flax module
class Sin(nn.Module):
    def __call__(self, x):
        return jnp.sin(x)  # Apply the sine function

# Define an ELU activation function as a Flax module
class Elu(nn.Module):
    def __call__(self, x):
        return nn.elu(x)  # Apply the ELU activation function

# Define a GLU (Gated Linear Unit) activation function as a Flax module
class GLU(nn.Module):
    def __call__(self, x):
        return nn.glu(x)  # Apply the GLU activation function

# Define a Layer Normalized ReLU activation function as a Flax module
class LayerNormedReLU(nn.Module):
    @nn.compact  # Indicates that this module can be used in a compact way
    def __call__(self, x):
        return nn.LayerNorm()(nn.relu(x))  # Apply ReLU followed by Layer Normalization

# Define a ReLU activation function normalized by its maximum value
class ReLUOverMax(nn.Module):
    def __call__(self, x):
        act = nn.relu(x)  # Apply ReLU activation
        return act / (jnp.max(act) + 1e-6)  # Normalize by the maximum value with a small epsilon to avoid division by zero

# Dictionary mapping activation function names to their corresponding classes
activation_fn = {
    # Unbounded activation functions
    "relu": ReLU,
    "elu": Elu,
    "glu": GLU,
    # Bounded activation functions
    "tanh": Tanh,
    "sin": Sin,
    "relu6": ReLU6,
    # Unbounded activation functions with normalizers
    "layernormed_relu": LayerNormedReLU,  # ReLU with layer normalization
    "relu_over_max": ReLUOverMax,  # ReLU normalized by its maximum value
}
