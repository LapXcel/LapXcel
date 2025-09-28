import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax import optim

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
    latents: int = 64

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()
    
    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def model():
    return VAE(latents=LATENTS)


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


@jax.jit
def train_step(optimizer, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)

        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss, recon_x

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    _, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer


rng = random.PRNGKey(0)
rng, key = random.split(rng)

init_data = jnp.ones((BATCH_SIZE, 784), jnp.float32)
params = model().init(key, init_data, rng)['params']

optimizer = optim.Adam(learning_rate=LEARNING_RATE).create(params)
optimizer = jax.device_put(optimizer)

rng, z_key, eval_rng = random.split(rng, 3)
z = random.normal(z_key, (64, LATENTS))

steps_per_epoch = 50000 // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    for _ in range(steps_per_epoch):
        batch = next(train_ds)
        rng, key = random.split(rng)
        optimizer = train_step(optimizer, batch, key)