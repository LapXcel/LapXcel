import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle
from flax.training import train_state
import optax
import cv2

class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(512)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mean, logvar


class Decoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(20 * 40 * 256)(z)
        z = nn.relu(z)
        z = z.reshape((-1, 20, 40, 256))

        z = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.relu(z)
        z = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.sigmoid(z)

        reconstruction = z.reshape((z.shape[0], -1))
        return reconstruction

class VAE():
    def reparameterize(self, key, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, logvar.shape)
        return mean + eps * std


    def compute_loss(self, params, batch, key):
        mean, logvar = self.encoder.apply({'params': params['encoder']}, batch)
        z = self.reparameterize(key, mean, logvar)
        reconstruction = self.decoder.apply({'params': params['decoder']}, z)

        recon_loss = jnp.mean(jnp.square(batch.reshape(batch.shape[0], -1) - reconstruction))
        kl_loss = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))

        return recon_loss + kl_loss


    def update_model(self, batch, key):
        def loss_fn(params):
            return self.compute_loss(params, batch, key)

        grads = jax.grad(loss_fn)(self.state.params)
        return self.state.apply_gradients(grads=grads)


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.state.params, f)


    def load_model(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        return self.state.replace(params=params)


    def encode(self, image_np):
        """
        Encode a single preprocessed image to the latent space.

        Args:
            image_np: numpy array of shape (160, 640, 3), preprocessed (BGR, resized, normalized).
            rng_key: JAX PRNGKey for sampling latent vector.

        Returns:
            latent_mean: latent mean vector, shape (latent_dim,)
            latent_sample: latent vector sampled using reparameterization trick, shape (latent_dim,)
        """
        # Add batch dimension
        batch = jnp.expand_dims(image_np, axis=0)  # (1, 320, 640, 3)

        # Get mean and logvar from encoder
        mean, logvar = self.encoder.apply({'params': self.state.params['encoder']}, batch)
        mean = mean[0]    # remove batch dimension
        logvar = logvar[0]

        # Sample from latent distribution with reparameterization trick
        z = self.reparameterize(self.rng_key, mean, logvar)

        return [mean, z]
    

    def decode(self, z):
        """
        Decode a latent vector z to an image reconstruction.

        Args:
            z: latent vector of shape (latent_dim,) or (batch_size, latent_dim)

        Returns:
            reconstruction: decoded image tensor, shape (batch_size, 320, 640, 3)
        """
        if z.ndim == 1:
            z = jnp.expand_dims(z, axis=0)

        flat_reconstruction = self.decoder.apply({'params': self.state.params['decoder']}, z)

        # Reshape output to image shape (batch_size, 320, 640, 3)
        reconstruction = flat_reconstruction.reshape((1, 320, 640, 3))
        return reconstruction
    
    def display_image(self, image):
        """
        Display a single image using matplotlib.

        Args:
            image: decoded image numpy array of shape (320, 640, 3)
        """
        # Convert from JAX DeviceArray to numpy and clip values if needed
        # Convert from JAX DeviceArray to numpy and clip values to [0, 1]
        # image_np = jnp.clip(image, 0, 1)
        # image_np = jax.device_get(image_np)  # move to host memory
        # image_np = (image_np * 255).astype('uint8')  # scale to 0-255 for OpenCV

        # # Convert RGB to BGR (OpenCV uses BGR)
        # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Display image in a window
        cv2.imshow('Decoded Image', image)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()