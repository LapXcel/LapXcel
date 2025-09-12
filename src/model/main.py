from ac_socket import ACSocket
from environment import Env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Car data (Ferrari SF70H)
    max_speed = 320.0

    model = None
    sock = ACSocket()
    print("Waiting for socket connection...")

    with sock.connect() as conn:

        # Set the socket in the environment
        # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
        env = Env(max_speed=max_speed)
        env.unwrapped.set_sock(sock)

        model = SAC(policy="MlpPolicy",
                    env=env,
                    verbose=1,
                    normalize= True,
                    n_envs=2,
                    n_timesteps=float 1e6,
                    learning_rate=float 7.3e-4,
                    buffer_size=300000,
                    batch_size=256,
                    ent_coef='auto',
                    gamma=0.99,
                    tau=0.02,
                    train_freq=8,
                    gradient_steps=10,
                    learning_starts=1000,
                    use_sde=True,
                    use_sde_at_warmup=True,
                    device="auto",
                    tensorboard_log="/app/logs/")
        # model = SAC.load("/app/checkpoints/sac_lapxcel_13000_steps", env, device="cuda")
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="/app/checkpoints/", name_prefix="sac_lapxcel")
        model.learn(total_timesteps=10000000000, callback=checkpoint_callback)
        model.save("/app/sac_lapxcel")

        env.controller.on_close()
        conn.close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()