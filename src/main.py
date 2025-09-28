from sbx import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.vae_env import ACVAEEnv
# from vae.vae import VAE

from config import FRAME_SKIP, MIN_THROTTLE, MAX_THROTTLE, Z_SIZE

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # vae = VAE(z_size=Z_SIZE)
    # vae.load_model("/app/vae/vae_trained_params.pkl")
    env = ACVAEEnv(frame_skip=FRAME_SKIP, vae=None, min_throttle=MIN_THROTTLE, max_throttle=MAX_THROTTLE, verbose=False, n_stack=4)

    model = TQC(policy="MlpPolicy",
                env=env,
                device="cuda",
                # tensorboard_log="/app/logs/tensorboard/",
                buffer_size=30_000,   # reasonable for 13GB GPU
                batch_size=64,        # can handle larger batches on 13GB
                learning_starts=1_000, # give buffer some data before learning
                train_freq=1,          # update every step
                gradient_steps=1,
                verbose=1)
    # model = TQC.load("/app/tqc_current", env)
    # model = SAC.load("/app/checkpoints/sac_lapxcel_13000_steps", env, device="cuda")
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="/app/logs/checkpoints/", name_prefix="tqc_lapxcel")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback, log_interval=1)
    model.save("/app/logs/models/tqc_lapxcel")


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()