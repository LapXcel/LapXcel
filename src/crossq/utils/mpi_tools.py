from mpi4py import MPI  # Import the MPI module for parallel processing
import os  # Import os module for interacting with the operating system
import subprocess  # Import subprocess module for running external commands
import sys  # Import sys module for system-specific parameters and functions
import numpy as np  # Import numpy for numerical operations


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of processes to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    # If n is less than or equal to 1, no need to fork
    if n <= 1:
        return
    
    # Check if the environment variable "IN_MPI" is not set
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()  # Copy the current environment variables
        # Update environment variables to restrict threading and indicate MPI
        env.update(
            MKL_NUM_THREADS="1",  # Set MKL to use a single thread
            OMP_NUM_THREADS="1",  # Set OpenMP to use a single thread
            IN_MPI="1"  # Set a flag to indicate that we are in MPI mode
        )
        
        # Prepare the command to launch MPI processes
        args = ["mpirun", "-np", str(n)]  # Specify number of processes
        if bind_to_core:
            args += ["-bind-to", "core"]  # Optionally bind to CPU cores
        args += [sys.executable] + sys.argv  # Add the script to run and its arguments
        
        # Execute the command to launch the MPI processes
        subprocess.check_call(args, env=env)
        sys.exit()  # Terminate the original process


def msg(m, string=''):
    """
    Print a message from the current MPI process.

    Args:
        m: The message to be printed.
        string: An optional string to include in the message.
    """
    # Print the rank of the current process and the message
    print(('Message from %d: %s \t ' %
          (MPI.COMM_WORLD.Get_rank(), string)) + str(m))


def proc_id():
    """Get the rank of the calling process."""
    return MPI.COMM_WORLD.Get_rank()  # Return the rank of the current MPI process


def allreduce(*args, **kwargs):
    """Perform a collective reduction operation across all processes."""
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)  # Call MPI's Allreduce function


def num_procs():
    """Count the number of active MPI processes."""
    return MPI.COMM_WORLD.Get_size()  # Return the total number of processes


def broadcast(x, root=0):
    """Broadcast a message from the root process to all other processes."""
    MPI.COMM_WORLD.Bcast(x, root=root)  # Use MPI's broadcast function


def mpi_op(x, op):
    """
    Perform a specified MPI operation (like sum or min) on the input.

    Args:
        x: Input data (can be a scalar or array).
        op: The MPI operation to perform.

    Returns:
        The result of the MPI operation.
    """
    # Check if x is a scalar and convert to array if necessary
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    
    x = np.asarray(x, dtype=np.float32)  # Ensure x is a float32 array
    buff = np.zeros_like(x, dtype=np.float32)  # Create a buffer for the result
    
    allreduce(x, buff, op=op)  # Perform the reduction operation
    return buff[0] if scalar else buff  # Return the result, handling scalars


def mpi_sum(x):
    """Compute the sum of x across all MPI processes."""
    return mpi_op(x, MPI.SUM)  # Use mpi_op to perform sum operation


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()  # Calculate average by dividing sum by number of processes


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.

    Returns:
        A tuple containing mean, std, and optionally min and max.
    """
    x = np.array(x, dtype=np.float32)  # Convert input to float32 array
    # Compute global sum and count of samples
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n  # Calculate mean

    # Compute global sum of squared differences for std
    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # Calculate global standard deviation

    if with_min_and_max:
        # Compute global min and max if requested
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max  # Return all statistics
    return mean, std  # Return mean and std only
