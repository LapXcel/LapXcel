from ac_socket import ACSocket
from environment import Env
from stable_baselines3 import SAC

def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Car data (Ferrari SF70H)
    max_speed = 320.0

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    env = Env(max_speed=max_speed)

    # Establish a socket connection
    sock = ACSocket()
    print("Waiting for socket connection...")
    with sock.connect() as conn:

        # Set the socket in the environment
        env.unwrapped.set_sock(sock)

        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save("sac_lapxcel")


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()