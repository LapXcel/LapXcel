import argparse
import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np



os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['WANDB_DIR'] = '/tmp'

parser = argparse.ArgumentParser()
parser.add_argument("-env",         type=str, required=False, default="HumanoidStandup-v4", help="Set Environment.")
parser.add_argument("-algo",        type=str, required=True, default='sac', choices=['crossq', 'sac', 'redq', 'droq', 'td3'], help="algorithm to use (essentially a named hyperparameter set for the base SAC algorithm)")
parser.add_argument("-seed",        type=int, required=False, default=1, help="Set Seed.")
parser.add_argument("-log_freq",    type=int, required=False, default=300, help="how many times to log during training")

parser.add_argument('-wandb_entity', type=str, required=False, default=None, help='your wandb entity name')
parser.add_argument('-wandb_project', type=str, required=False, default='crossQ', help='wandb project name')
parser.add_argument("-wandb_mode",    type=str, required=False, default='disabled', choices=['disabled', 'online'], help="enable/disable wandb logging")
parser.add_argument("-eval_qbias",    type=int, required=False, default=0, choices=[0,1], help="enable/diasble q bias evaluation (expensive; experiments will run much slower)")

parser.add_argument("-adam_b1",           type=float, required=False, default=0.5, help="adam b1 hyperparameter")
parser.add_argument("-bn",                type=float, required=False, default=False,  choices=[0,1], help="Use batch norm layers in the actor and critic networks")
parser.add_argument("-bn_momentum",       type=float, required=False, default=0.99, help="batch norm momentum parameter")
parser.add_argument("-bn_mode",           type=str,   required=False, default='brn_actor', help="batch norm mode (bn / brn / brn_actor). brn_actor also uses batch renorm in the actor network")
parser.add_argument("-critic_activation", type=str,   required=False, default='relu', help="critic activation function")
parser.add_argument("-crossq_style",      type=float, required=False, default=1,choices=[0,1], help="crossq style joint forward pass through critic network")
parser.add_argument("-dropout",           type=int,   required=False, default=0, choices=[0,1], help="whether to use dropout for SAC")
parser.add_argument("-ln",                type=float, required=False, default=False, choices=[0,1], help="layernorm in critic network")
parser.add_argument("-lr",                type=float, required=False, default=1e-3, help="actor and critic learning rate")
parser.add_argument("-n_critics",         type=int,   required=False, default=2, help="number of critics to use")
parser.add_argument("-n_neurons",         type=int,   required=False, default=256, help="number of neurons for each critic layer")
parser.add_argument("-policy_delay",      type=int,   required=False, default=1, help="policy is updated after this many critic updates")
parser.add_argument("-tau",               type=float, required=False, default=0.005, help="target network averaging")
parser.add_argument("-utd",               type=int,   required=False, default=1, help="number of critic updates per env step")
parser.add_argument("-total_timesteps",   type=int,   required=False, default=5e6, help="total number of training steps")

parser.add_argument("-bnstats_live_net",  type=int,   required=False, default=0,choices=[0,1], help="use bn running statistics from live network within the target network")

experiment_time = time.time()
args = parser.parse_args()

seed = args.seed
args.algo = str.lower(args.algo)
args.bn = bool(args.bn)
args.crossq_style = bool(args.crossq_style)
args.tau = float(args.tau) if not args.crossq_style else 1.0
args.bn_momentum = float(args.bn_momentum) if args.bn else 0.0
dropout_rate, layer_norm = None, False
policy_q_reduce_fn = 0
net_arch = {'pi': [256, 256], 'qf': [args.n_neurons, args.n_neurons]}

total_timesteps = int(args.total_timesteps)
eval_freq = max(5_000_000 // args.log_freq, 1)

if 'dm_control' in args.env:
    total_timesteps = {
        'dm_control/reacher-easy'     : 100_000,
        'dm_control/reacher-hard'     : 100_000,
        'dm_control/ball_in_cup-catch': 200_000,
        'dm_control/finger-spin'      : 500_000,
        'dm_control/fish-swim'        : 5_000_000,
        'dm_control/humanoid-stand'   : 5_000_000,
    }[args.env]
    eval_freq = max(total_timesteps // args.log_freq, 1)

td3_mode = False

if args.algo == 'droq':
    dropout_rate = 0.01
    layer_norm = True
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 2
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'DroQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'redq':
    policy_q_reduce_fn = jax.numpy.mean
    args.n_critics = 10
    # args.adam_b1 = 0.9  # adam default
    args.adam_b2 = 0.999  # adam default
    args.policy_delay = 20
    args.utd = 20
    group = f'REDQ_{args.env}_bn({args.bn})_ln{(args.ln)}_xqstyle({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_Adam({args.adam_b1})_Q({net_arch["qf"][0]})'

elif args.algo == 'td3':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    td3_mode = True
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'TD3_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'sac':
    # With the right hyperparameters, this here can run all the above algorithms
    # and ablations.
    layer_norm = args.ln
    if args.dropout: 
        dropout_rate = 0.01
    group = f'SAC_{args.env}_bn({args.bn}/{args.bn_momentum}/{args.bn_mode})_ln{(args.ln)}_xq({args.crossq_style}/{args.tau})_utd({args.utd}/{args.policy_delay})_A{args.adam_b1}_Q({net_arch["qf"][0]})_l{args.lr}'

elif args.algo == 'crossq':
    args.adam_b1 = 0.5
    args.policy_delay = 3
    args.n_critics = 2
    args.utd = 1                    # nice
    net_arch["qf"] = [2048, 2048]   # wider critics
    args.bn = True                  # use batch norm
    args.bn_momentum = 0.99
    args.crossq_style = True        # with a joint forward pass
    args.tau = 1.0                  # without target networks
    group = f'CrossQ_{args.env}'

else:
    raise NotImplemented

args_dict = vars(args)
args_dict.update({
    "dropout_rate": dropout_rate,
    "layer_norm": layer_norm
})

print(args_dict)

 