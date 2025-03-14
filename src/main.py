import os

import numpy as np
from ac_socket import ACSocket
from crossq.environment import Env
from crossq.crossq import SAC
import optax
import jax
from crossq.utils.utils import *
import time
from crossq.actor_critic_evaluation_callback import CriticBiasCallback, EvalCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


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
    with sock.connect() as conn:

        # Set the socket in the environment
        env.unwrapped.set_sock(sock)

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
        group = f'CrossQ_AssetoCorsa'
        experiment_time = time.time()
        eval_freq = 1
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

        # Create log dir where evaluation results will be saved
        eval_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/eval/"
        qbias_log_dir = f"./eval_logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/qbias/"
        os.makedirs(eval_log_dir, exist_ok=True)
        os.makedirs(qbias_log_dir, exist_ok=True)

        # Create callback that evaluates agent
        eval_callback = EvalCallback(
            make_vec_env(args.env, n_envs=1, seed=seed),
            jax_random_key_for_seeds=args.seed,
            best_model_sav=eval_log_dir, eval_freq=eval_freq,
            n_eval_episodes=1, deterministic=True, render=False
        )

        # Callback that evaluates q bias according to the REDQ paper.
        q_bias_callback = CriticBiasCallback(
            make_vec_env(args.env, n_envs=1, seed=seed), 
            jax_random_key_for_seeds=args.seed,
            best_model_save_path=None,
            log_path=qbias_log_dir, eval_freq=eval_freq,
            n_eval_episodes=1, render=False
        )

        callback_list = CallbackList([eval_callback, q_bias_callback, WandbCallback(verbose=0,)])

        # Run the training loop
        agent.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback_list)


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()