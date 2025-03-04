import os

import numpy as np
from ac_socket import ACSocket
from gymnasium.wrappers import TimeLimit
from crossq.environment import Env
from crossq.utils.logx import colorize
from crossq.crossq import SAC


def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Car data (Ferrari SF70H)
    max_speed = 320.0

    # Initialize the environment, max_episode_steps is the maximum amount of steps before the episode is truncated
    env = TimeLimit(Env(max_speed=max_speed), max_episode_steps=1000)

    # Initialize the agent
    args = {
        "adam_b1": 0.5,
        "policy_delay": 3,
        "n_critics": 2,
        "utd": 1,                    # nice
        "net_arch": {"qf": [2048, 2048]},   # wider critics
        "bn": True,                # use batch norm
        "bn_momentum": 0.99,
        "crossq_style": True,        # with a joint forward pass
        "tau": 1.0,                  # without target networks
        "group": f'CrossQ_{args.env}'
    }

    agent = SAC(env, exp_name, load_path, **hyperparams)

    # Establish a socket connection
    sock = ACSocket()
    with sock.connect() as conn:

        # Set the socket in the environment
        env.unwrapped.set_sock(sock)

        # Run the training loop
        agent.train()


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()