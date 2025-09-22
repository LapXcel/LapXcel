import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle
from flax.training import train_state
import optax

class VAE():
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
        latent_dim: int

        @nn.compact
        def __call__(self, z):
            x = nn.Dense(20 * 40 * 256)(z)
            x = nn.relu(x)
            x = x.reshape((-1, 20, 40, 256))

            x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
            x = nn.relu(x)
            x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)

            reconstruction = x.reshape((x.shape[0], -1))
            return reconstruction

    def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001,
                 kl_tolerance=0.5, beta=1.0):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.beta = beta
        self.encoder = self.Encoder(z_size)
        self.decoder = self.Decoder(z_size)

        # Initialize parameters here
        self.rng_key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 320, 640, 3))  # batch size 1, your input size

        encoder_params = self.encoder.init(self.rng_key, dummy_input)['params']
        decoder_params = self.decoder.init(self.rng_key, jnp.ones((1, z_size)))['params']

        self.params = {'encoder': encoder_params, 'decoder': decoder_params}

        tx = optax.adam(learning_rate=1e-3)
        self.state = train_state.TrainState.create(apply_fn=None, params=self.params, tx=tx)

    def reparameterize(self, key, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, shape=std.shape)
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
        # z = self.reparameterize(self.rng_key, mean, logvar)

        return [mean]