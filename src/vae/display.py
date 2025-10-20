from vae import VAE
import cv2
import numpy as np

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

# cv2.imshow('Decoded Image', load_and_preprocess_image("./imgs/scene00001.jpeg"))
# cv2.waitKey(0)  # Wait indefinitely until a key is pressed
# cv2.destroyAllWindows()

vae = VAE()
vae.load_model("./vae_trained_params.pkl")
z = vae.encode(load_and_preprocess_image("./imgs/scene00001.jpeg"))[1]
print(z)
# vae.display_image(vae.decode(z)[0])
print(vae.decode(z)[0])