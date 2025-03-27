from typing import Any, Optional

import jax.numpy as jnp
import tensorflow_probability

# Import the TensorFlow Probability library with JAX support
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

class TanhTransformedDistribution(tfd.TransformedDistribution):  # Inherit from TransformedDistribution
    """
    A custom distribution that applies a Tanh transformation to another distribution.
    
    This class is useful for working with distributions that need to be squashed 
    to a certain range (like between -1 and 1), such as in the case of a Squashed Gaussian.
    The mode is defined after applying the Tanh transformation.
    
    Reference: https://github.com/ikostrikov/walk_in_the_park
    """

    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):  # Constructor
        """
        Initialize the TanhTransformedDistribution.

        Args:
            distribution (tfd.Distribution): The base distribution to be transformed.
            validate_args (bool): Whether to validate distribution arguments. Default is False.
        """
        # Call the parent class constructor with the distribution and a Tanh bijector
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        """
        Calculate the mode of the transformed distribution.

        Returns:
            jnp.ndarray: The mode of the distribution after applying the Tanh transformation.
        """
        # Apply the forward transformation of the bijector to the mode of the base distribution
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        """
        Define the properties of the distribution parameters.

        Args:
            cls: The class itself.
            dtype (Optional[Any]): The data type of the parameters.
            num_classes (Optional): The number of classes for categorical distributions.

        Returns:
            dict: A dictionary describing the properties of the parameters.
        """
        # Get the parameter properties from the parent class
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        # Remove the 'bijector' property as it is not needed
        del td_properties["bijector"]
        return td_properties
