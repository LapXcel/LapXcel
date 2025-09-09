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

    model = None
    sock = ACSocket()
    print("Waiting for socket connection...")
    try:
        with sock.connect() as conn:
            env.unwrapped.set_sock(sock)
            model = SAC("MlpPolicy", env, verbose=1, device="cuda")
            model.learn(total_timesteps=100000)
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        if model is not None:
            try:
                model.save("./sac_lapxcel")
                print("Model saved successfully.")
            except Exception as save_e:
                print(f"Failed to save the model: {save_e}")
        try:
            env.controller.on_close()
        except Exception as close_e:
            print(f"Error during controller close: {close_e}")

        try:
            conn.close()
        except Exception as close_conn_e:
            print(f"Error closing socket connection: {close_conn_e}")


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()