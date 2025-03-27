from ac_socket import ACSocket  # Import the ACSocket class for socket communication
from crossq.environment import Env  # Import the Env class for the environment setup
from crossq.crossq import SAC  # Import the SAC class (Soft Actor-Critic) for the agent
import optax  # Import Optax for optimization
import jax  # Import JAX for numerical computing
from crossq.utils.utils import *  # Import utility functions from the utils module
import time  # Import time module for timing operations


def main():
    """
    The main function of the standalone application.
    It will initialize the environment and the agent, and then run the training loop.
    """
    # Define the maximum speed for the car (Ferrari SF70H)
    max_speed = 320.0

    # Initialize the environment with the specified maximum speed
    # max_episode_steps is the maximum amount of steps before the episode is truncated
    env = Env(max_speed=max_speed)

    # Establish a socket connection for communication
    sock = ACSocket()
    with sock.connect() as conn:  # Use a context manager to ensure the socket is properly closed

        # Set the socket in the environment for communication
        env.unwrapped.set_sock(sock)

        # Initialize the agent with specified hyperparameters
        args = {
            'algo': 'crossq',  # The algorithm to be used
            'seed': 1,  # Random seed for reproducibility
            'log_freq': 300,  # Frequency of logging
            'wandb_entity': None,  # Weights & Biases entity (if using W&B)
            'wandb_project': 'crossQ',  # W&B project name
            'wandb_mode': 'disabled',  # Mode for W&B logging
            'eval_qbias': 0,  # Evaluation Q bias
            'adam_b1': 0.5,  # Adam optimizer beta1 parameter
            'bn': True,  # Whether to use batch normalization
            'bn_momentum': 0.99,  # Momentum for batch normalization
            'bn_mode': 'brn_actor',  # Batch normalization mode
            'critic_activation': 'relu',  # Activation function for the critic
            'crossq_style': True,  # Whether to use CrossQ style
            'dropout': 0,  # Dropout rate
            'ln': False,  # Whether to use layer normalization
            'lr': 0.001,  # Learning rate
            'n_critics': 2,  # Number of critic networks
            'n_neurons': 256,  # Number of neurons in each layer
            'policy_delay': 3,  # Delay for policy updates
            'tau': 1.0,  # Soft update coefficient
            'utd': 1,  # Update the target network every 'utd' steps
            'total_timesteps': 50000.0,  # Total timesteps for training
            'bnstats_live_net': 0,  # Batch normalization stats for the live network
            'dropout_rate': None,  # Dropout rate (if any)
            'layer_norm': False  # Whether to use layer normalization
        }

        # Set up experiment parameters
        seed = 1  # Seed for random number generation
        group = f'CrossQ_AssetoCorsa'  # Experiment group name
        experiment_time = time.time()  # Record the start time of the experiment

        # Initialize the SAC agent with the specified parameters
        agent = SAC(
            "MultiInputPolicy",  # The policy architecture to use
            env,  # The environment instance
            policy_kwargs=dict({
                'activation_fn': activation_fn[args["critic_activation"]],  # Activation function for the policy
                'layer_norm': False,  # Disable layer normalization
                'batch_norm': bool(args["bn"]),  # Enable/disable batch normalization
                'batch_norm_momentum': float(args["bn_momentum"]),  # Batch normalization momentum
                'batch_norm_mode': args["bn_mode"],  # Batch normalization mode
                'dropout_rate': None,  # Dropout rate
                'n_critics': args["n_critics"],  # Number of critics
                'net_arch': {'pi': [256, 256], 'qf': [2048, 2048]},  # Network architecture for policy and Q-function
                'optimizer_class': optax.adam,  # Optimizer class to use
                'optimizer_kwargs': dict({
                    'b1': args["adam_b1"],  # First moment estimate decay rate
                    'b2': 0.999  # Second moment estimate decay rate (default)
                })
            }),
            gradient_steps=args["utd"],  # Number of gradient steps per update
            policy_delay=args["policy_delay"],  # Policy update delay
            crossq_style=bool(args["crossq_style"]),  # Use CrossQ style
            td3_mode=False,  # Disable TD3 mode
            use_bnstats_from_live_net=bool(args["bnstats_live_net"]),  # Use batch normalization stats from live network
            policy_q_reduce_fn=jax.numpy.min,  # Function to reduce Q-values
            learning_starts=5000,  # Steps before starting to learn
            learning_rate=args["lr"],  # Learning rate for the Q-function
            qf_learning_rate=args["lr"],  # Learning rate for the Q-function
            tau=args["tau"],  # Soft update coefficient
            gamma=0.99,  # Discount factor
            verbose=0,  # Verbosity level
            buffer_size=1_000_000,  # Size of the replay buffer
            seed=seed,  # Seed for random number generation
            stats_window_size=1,  # Don't smooth the episode return stats over time
            tensorboard_log=f"logs/{group + 'seed=' + str(seed) + '_time=' + str(experiment_time)}/",  # Path for TensorBoard logs
        )

        # Run the training loop for the specified number of timesteps
        agent.learn(total_timesteps=args["total_timesteps"], progress_bar=True)


if __name__ == "__main__":
    """
    Run the main function when the script is executed.
    """
    main()  # Call the main function to start the program
