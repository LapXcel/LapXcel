from sbx import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from envs.vae_env import ACVAEEnv
import wandb
from wandb.integration.sb3 import WandbCallback
# from vae.vae import VAE

from config import FRAME_SKIP, MIN_THROTTLE, MAX_THROTTLE, Z_SIZE

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # vae = VAE(z_size=Z_SIZE)
    # vae.load_model("/app/vae/vae_trained_params.pkl")
    env = ACVAEEnv(frame_skip=FRAME_SKIP, vae=None, min_throttle=MIN_THROTTLE, max_throttle=MAX_THROTTLE, verbose=False, n_stack=4, n_command_history=2)

    config={
        "policy": "MlpPolicy",
        "buffer_size": 30000,
        "batch_size": 64,
        "learning_starts": 10000,
        "train_freq": 1,
        "gradient_steps": 1,
        "env": env,
        "tensorboard_log": "/app/logs/tensorboard"
        # Add other hyperparameters as needed
    }

    model = TQC(**config)
    # model = TQC.load("/app/tqc_lapxcel_320000_steps", env)

    wandbConfig = {
        "policy": "MlpPolicy",  # or just "MlpPolicy" if static
        "algo": "TQC",
        "buffer_size": model.buffer_size,
        "batch_size": model.batch_size,
        "learning_starts": model.learning_starts,
        "train_freq": model.train_freq,
        "gradient_steps": model.gradient_steps,
        "env": env.spec.id if env.spec else "CustomEnv",
    }

    run = wandb.init(
        project="Lapxcel",
        config=wandbConfig,
        sync_tensorboard=True, # uploads SB3's TensorBoard metrics
        monitor_gym=False,      # uploads videos of agents
        save_code=False         # optional
    )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="/app/logs/checkpoints/", name_prefix="tqc_lapxcel")
    callbacks = CallbackList([
        checkpoint_callback,
        WandbCallback(
            gradient_save_freq=1000,
            model_save_path="/app/logs/models/",
            verbose=2,
        )
    ])

    # model = SAC.load("/app/checkpoints/sac_lapxcel_13000_steps", env, device="cuda")
    model.learn(total_timesteps=1_000_000, callback=callbacks, log_interval=1)
    model.save("/app/logs/models/tqc_lapxcel")
    run.finish()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()
