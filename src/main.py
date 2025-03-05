import os

import numpy as np
from ac_socket import ACSocket
from gymnasium.wrappers import TimeLimit
from crossq.environment import Env
from crossq.utils.logx import colorize
from crossq.crossq import SAC
import optax
import jax
from crossq.utils.utils import *
import time


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
            'algo': 'crossq',
            'seed': 1,
            'log_freq': 300,
            'wandb_entity': None,
            'wandb_project': 'crossQ',
            'wandb_mode': 'disabled',
            'eval_qbias': 0,
            'adam_b1': 0.5,
            'bn': True,
            'bn_momentum': 0.99,
            'bn_mode': 'brn_actor',
            'critic_activation': 'relu',
            'crossq_style': True,
            'dropout': 0,
            'ln': False,
            'lr': 0.001,
            'n_critics': 2,
            'n_neurons': 256,
            'policy_delay': 3,
            'tau': 1.0,
            'utd': 1,
            'total_timesteps': 5000000.0,
            'bnstats_live_net': 0,
            'dropout_rate': None,
            'layer_norm': False
        }

    seed = 1
    group = f'CrossQ_{args.env}'
    experiment_time = time.time()
    agent = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict({
            'activation_fn': activation_fn[args.critic_activation],
            'layer_norm': False,
            'batch_norm': bool(args.bn),
            'batch_norm_momentum': float(args.bn_momentum),
            'batch_norm_mode': args.bn_mode,
            'dropout_rate': None,
            'n_critics': args.n_critics,
            'net_arch': {'pi': [256, 256], 'qf': [2048, 2048]},
            'optimizer_class': optax.adam,
            'optimizer_kwargs': dict({
                'b1': args.adam_b1,
                'b2': 0.999 # default
            })
        }),
        gradient_steps=args.utd,
        policy_delay=args.policy_delay,
        crossq_style=bool(args.crossq_style),
        td3_mode=False,
        use_bnstats_from_live_net=bool(args.bnstats_live_net),
        policy_q_reduce_fn=jax.numpy.min,
        learning_starts=5000,
        learning_rate=args.lr,
        qf_learning_rate=args.lr,
        tau=args.tau,
        gamma=0.99,
        verbose=0,
        buffer_size=1_000_000,
        seed=seed,
        stats_window_size=1,  # don't smooth the episode return stats over time
        tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",
    )

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