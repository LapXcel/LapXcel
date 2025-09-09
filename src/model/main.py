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

        model = SAC("MlpPolicy", env, verbose=1, device="cuda")
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='/app/checkpoints/', name_prefix='sac_lapxcel')
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
        model.save("/app/sac_lapxcel")

        env.controller.on_close()
        conn.close()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()