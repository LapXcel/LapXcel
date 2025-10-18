import optuna
from sbx import TQC
from envs.vae_env import ACVAEEnv
import wandb
from wandb.integration.sb3 import WandbCallback

from config import FRAME_SKIP, MIN_THROTTLE, MAX_THROTTLE, Z_SIZE

def optimize_tqc(trial, env):
    # Sample hyperparameters from trial
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 30000, 50000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 10)

    # Initialize model with sampled hyperparameters
    model = TQC(
        "MlpPolicy",
        env,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=500,
        tensorboard_log="/app/logs/tensorboard"
    )

    # # Optional: initialize wandb for logging (optional for each trial)
    # run = wandb.init(
    #     project="Lapxcel",
    #     config={
    #         "buffer_size": buffer_size,
    #         "batch_size": batch_size,
    #         "learning_rate": learning_rate,
    #         "train_freq": train_freq,
    #         "gradient_steps": gradient_steps,
    #         "algo": "TQC",
    #     },
    #     reinit=True,
    #     save_code=False
    # )

    # # Set up callbacks for Wandb
    # callbacks = [WandbCallback(gradient_save_freq=1000, model_save_path="/app/logs/models/", verbose=0)]

    # Train for a smaller number of timesteps per trial (adjust as needed)
    model.learn(total_timesteps=1000)

    # Evaluate the trained model performance on env (e.g., average episode reward)
    episode_rewards = []
    for _ in range(5):  # Run 5 episodes for evaluation
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Trial params: {trial.params}, Mean Reward: {mean_reward}")
    # run.finish()

    return mean_reward

if __name__ == "__main__":
    # Create environment instance (single env, no parallel)
    env = ACVAEEnv(frame_skip=FRAME_SKIP, vae=None, min_throttle=MIN_THROTTLE, max_throttle=MAX_THROTTLE, verbose=False, n_stack=4, n_command_history=2)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_tqc(trial, env), n_trials=20)
    print(f"Best trial: {study.best_trial.params}, Reward: {study.best_trial.value}")
