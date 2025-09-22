"""Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using PPO implementation from Stable-Baselines3
on a Gymnasium environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python optimize_ppo.py

"""

from typing import Any

import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sbx import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import torch
import torch.nn as nn


N_TRIALS = 500
N_STARTUP_TRIALS = 10
N_EVALUATIONS = 2
N_TIMESTEPS = 40_000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 10

N_ENVS = 1

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


def sample_tqc_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sampler for TQC hyperparameters."""
    # Discount factor
    gamma = trial.suggest_float("gamma", 0.975, 0.995)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.002, log=True)
    # Initial exploration rate (entropy coefficient in the SAC loss)
    ent_coef_init = trial.suggest_float("ent_coef_init", 0.001, 0.02, log=True)
    # From 2^7=128 to 2^12 = 4096, the mini-batch size
    batch_size_pow = trial.suggest_int("batch_size_pow", 7, 12, log=True)
    # How big should should the actor and critic networks be
    # net_arch = trial.suggest_categorical("net_arch", ["default", "simba", "large"])
    # I'm using integers to be able to use CMA-ES,
    # "default" is [256, 256], "large" is [512, 256, 128]
    net_arch_complexity = trial.suggest_int("net_arch_complexity", 0, 3)
    # From 1 to 8 (how often should we update the networks, every train_freq steps in the env)
    train_freq_pow = trial.suggest_int("train_freq_pow", 0, 3)
    # From 1 to 1024 (how many gradient steps by step in the environment)
    gradient_steps_pow = trial.suggest_int("gradient_steps_pow", 0, 10)
    # From 1 to 32 (the policy delay parameter, similar to TD3 update)
    policy_delay_pow = trial.suggest_int("policy_delay_pow", 0, 5)
    # Polyak coeff (soft update of the target network)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)
    
    # Display true values
    trial.set_user_attr("batch_size", 2**batch_size_pow)
    trial.set_user_attr("gradient_steps", 2**gradient_steps_pow)
    trial.set_user_attr("policy_delay", 2**policy_delay_pow)
    trial.set_user_attr("train_freq", 2**train_freq_pow)
    # Note: to_hyperparams() does the convertions between sampled value and expected value
    # Ex: converts batch_size_pow to batch_size
    # This is useful when replaying trials
    return to_hyperparams({
        "train_freq_pow": train_freq_pow,
        "gradient_steps_pow": gradient_steps_pow,
        "batch_size_pow": batch_size_pow,
        "tau": tau,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_delay_pow": policy_delay_pow,
        "ent_coef_init": ent_coef_init,
        "net_arch_complexity": net_arch_complexity,
    })


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.
    model = PPO(env=vec_env, **kwargs)
    # Create env used for evaluation.
    eval_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, multivariate=True)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")