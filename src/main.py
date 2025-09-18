from sbx import TQC
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.vae_env import ACVAEEnv

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    env = ACVAEEnv()

    model = TQC(policy="MlpPolicy",
                env=env,
                device="cuda",
                tensorboard_log="/app/logs/tensorboard/",
                buffer_size=200_000,   # reasonable for 13GB GPU
                batch_size=256,        # can handle larger batches on 13GB
                learning_starts=10_000, # give buffer some data before learning
                train_freq=1,          # update every step
                gradient_steps=1)
    # model = SAC.load("/app/checkpoints/sac_lapxcel_13000_steps", env, device="cuda")
    checkpoint_callback = CheckpointCallback(save_freq=25000, save_path="/app/logs/checkpoints/", name_prefix="tqc_lapxcel")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    model.save("/app/logs/models/tqc_lapxcel")


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()