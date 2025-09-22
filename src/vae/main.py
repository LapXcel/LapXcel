import os
from glob import glob
import cv2
import numpy as np
import jax
import jax.numpy as jnp

from vae import VAE  # Your VAE class file

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)
MARGIN_TOP = CAMERA_HEIGHT // 3
# Region Of Interest
# r = [margin_left, margin_top, width, height]
ROI = [0, MARGIN_TOP, CAMERA_WIDTH, CAMERA_HEIGHT - MARGIN_TOP]
IMAGE_WIDTH = ROI[2]
IMAGE_HEIGHT = ROI[3]

def load_and_preprocess_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)  # BGR format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency
    image = image[:, :, ::-1]  # RGB to BGR (as you want BGR)

    r = ROI
    image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # Normalize pixel values to [0, 1] float32 NumPy array
    image = image.astype(np.float32) / 255.0
    return image


def load_batch(image_files, batch_size, start_idx=0):
    batch_files = image_files[start_idx: start_idx + batch_size]
    batch = np.stack([load_and_preprocess_image(f) for f in batch_files])
    return batch


def training_loop(image_dir, vae, epochs=10, batch_size=16):
    image_files = glob(os.path.join(image_dir, '*.jpeg'))  # Adjust pattern if needed
    n_samples = len(image_files)
    rng = jax.random.PRNGKey(0)

    for epoch in range(epochs):
        np.random.shuffle(image_files)
        batch_jax = None
        key = None

        for i in range(0, n_samples, batch_size):
            batch = load_batch(image_files, batch_size, i)
            if batch.shape[0] < batch_size:
                continue

            batch_jax = jnp.array(batch)

            rng, key = jax.random.split(rng)
            vae.update_model(batch_jax, key)

        # Compute loss only if at least one batch processed
        if batch_jax is not None and key is not None:
            loss = vae.compute_loss(batch_jax, key)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}, no batches processed")

    return


def main():
    latent_dim = 64
    batch_size = 16
    epochs = 20
    image_dir = '/app/imgs'

    vae = VAE(z_size=latent_dim)  # Match your class argument name 'z_size'
    training_loop(image_dir, vae, epochs=epochs, batch_size=batch_size)

    vae.save_model('/app/vae_trained_params.pkl')
    print("Model saved as vae_trained_params.pkl")


if __name__ == '__main__':
    main()
