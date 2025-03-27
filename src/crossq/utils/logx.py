"""
Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)
"""

import json
import joblib
import numpy as np
import torch
import os.path as osp
import time
import atexit
import os
import warnings
from crossq.utils.mpi_tools import proc_id, mpi_statistics_scalar
from crossq.utils.serialization_utils import convert_json

# A dictionary mapping color names to their respective ANSI escape codes for terminal output
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function applies ANSI color codes to a string for terminal output.
    Originally written by John Schulman.
    """
    attr = []
    num = color2num[color]  # Get the color code
    if highlight:
        num += 10  # Adjust for highlighted colors
    attr.append(str(num))
    if bold:
        attr.append('1')  # Add bold attribute
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)  # Return the colorized string

class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If 
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them.
        """
        # Check if the process ID is 0 (main process)
        if proc_id() == 0:
            # Set output directory, create if it doesn't exist
            self.output_dir = output_dir or osp.join(
                os.getcwd(), "experiments", exp_name or "experiment_%i" % int(time.time()))
            if osp.exists(self.output_dir):
                # Warn if the output directory already exists
                print(colorize(
                    "Warning: Log dir %s already exists!" % self.output_dir, "yellow", bold=True))
                if input(colorize("Continue? [y/n] ", "gray")) != "y":
                    exit(-1)  # Exit if user chooses not to continue
            else:
                os.makedirs(self.output_dir)  # Create the directory
            # Open the output file for writing
            self.output_file = open(
                osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)  # Ensure file is closed on exit
            print(colorize("Logging data to %s" %
                  self.output_file.name, 'green', bold=True))
        else:
            # For non-main processes, set output_dir and output_file to None
            self.output_dir = None
            self.output_file = None
        self.first_row = True  # Flag to indicate the first row of logs
        self.log_headers = []  # List to store log headers
        self.log_current_row = {}  # Dictionary to store current log values
        self.exp_name = exp_name  # Store experiment name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))  # Print message in color

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)  # Add header on first log
        else:
            # Ensure that the key is already logged
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        # Ensure that the key is not already set in the current row
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val  # Store the value for the key

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way.

        Example use:
            logger = EpisodeLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)  # Convert config to JSON format
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name  # Add experiment name if provided
        if proc_id() == 0:
            output = json.dumps(config_json, separators=(
                ',', ':\t'), indent=4, sort_keys=True)  # Serialize config to JSON string
            print(colorize('Saving config', color='cyan', bold=True))  # Log saving action
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)  # Write JSON config to file

    def save_drive_data(self, e, speed, x_path, y_path, z_path):
        """
        Save the data from a driving episode to a JSON file.
        """
        # Convert float32 arrays to strings for JSON serialization
        speed = speed.astype(str).tolist()
        x_path = x_path.astype(str).tolist()
        y_path = y_path.astype(str).tolist()
        z_path = z_path.astype(str).tolist()

        # Create the file name for saving drive data
        fname = osp.join(self.output_dir, "drive_data.json")

        # If the JSON file already exists, load the existing data
        if osp.exists(fname):
            with open(fname, 'r') as file:
                data = json.load(file)  # Load existing data

            # Add new data to the episodes list
            data["episodes"].append({
                "episode": e,
                "speed": speed,
                "x_path": x_path,
                "y_path": y_path,
                "z_path": z_path
            })
        else:
            # Create a new dictionary for the first episode
            data = {
                "episodes": [{
                    "episode": e,
                    "speed": speed,
                    "x_path": x_path,
                    "y_path": y_path,
                    "z_path": z_path
                }]
            }

        # Save the updated data to the file
        with open(fname, 'w') as outfile:
            json.dump(data, outfile, indent=4, sort_keys=True)  # Write JSON data to file

    def load_config(self):
        """
        Load an experiment configuration from a saved JSON file.

        This function reads the "config.json" file saved by save_config
        and returns the configuration as a Python dictionary.

        Returns:
            dict: Experiment configuration loaded from the file.
        """
        config_path = osp.join(self.output_dir, "config.json")  # Path to config file
        if osp.exists(config_path):
            with open(config_path, 'r') as infile:
                config_json = json.load(infile)  # Load configuration data

            # Remove the 'exp_name' key if it was added during saving
            config = config_json.copy()
            if 'exp_name' in config:
                del config['exp_name']

            print(colorize('Loaded config:\n', color='cyan', bold=True))  # Log loading action
            print(json.dumps(config, separators=(
                ',', ':\t'), indent=4, sort_keys=True))  # Print loaded config

            return config  # Return the loaded configuration
        else:
            print(colorize('Config file not found.', color='red'))  # Warn if config file is missing
            return None

    def save_state(self, state_dict, save_env=False, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """
        if proc_id() == 0:  # Check if main process
            if save_env:
                # Save environment state if requested
                fname = 'env.pkl' if itr is None else 'env%d.pkl' % itr
                try:
                    joblib.dump(state_dict, osp.join(self.output_dir, fname))  # Save state to file
                except:
                    self.log('Warning: could not pickle state_dict.', color='red')  # Warn if saving fails
            if hasattr(self, 'pytorch_saver_elements'):
                self._pytorch_simple_save(itr)  # Save PyTorch model state

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save  # Store reference to model(s) to save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id() == 0:  # Check if main process
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"  # Ensure setup was called
            fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'  # Construct filename
            fname = osp.join(self.output_dir, fname)  # Full path for saving
            os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings during saving
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving).
                torch.save(self.pytorch_saver_elements, fname)  # Save the model(s)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        if proc_id() == 0:  # Check if main process
            vals = []  # List to hold logged values
            key_lens = [len(key) for key in self.log_headers]  # Get lengths of all keys
            max_key_len = max(15, max(key_lens))  # Determine max key length for formatting
            keystr = '%'+'%d' % max_key_len  # Format string for key
            fmt = "| " + keystr + "s | %15s |"  # Format string for output
            n_slashes = 22 + max_key_len  # Calculate total length for table
            print("-"*n_slashes)  # Print top border of the table
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")  # Get the value for the key
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val  # Format value for output
                print(fmt % (key, valstr))  # Print the key-value pair
                vals.append(val)  # Append value to list for saving
            print("-"*n_slashes, flush=True)  # Print bottom border of the table
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers)+"\n")  # Write headers to file
                self.output_file.write("\t".join(map(str, vals))+"\n")  # Write values to file
                self.output_file.flush()  # Ensure data is written to file
        self.log_current_row.clear()  # Clear current row for next iteration
        self.first_row = False  # Set first row flag to False for subsequent logs

class EpisodeLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over episodes.

    Typical use case: there is some quantity which is calculated many times
    throughout an episode, and at the end of the episode, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpisodeLogger, each time the quantity is calculated, you would
    use 

    .. code-block:: python

        episode_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpisodeLogger's state. Then at the end of the episode, you 
    would use 

    .. code-block:: python

        episode_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base Logger
        self.episode_dict = dict()  # Dictionary to store values for the current episode

    def store(self, **kwargs):
        """
        Save something into the episode_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.episode_dict.keys()):
                self.episode_dict[k] = []  # Initialize list for new key
            self.episode_dict[k].append(v)  # Append value to the list for the key

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the episode.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the episode.
        """
        if val is not None:
            super().log_tabular(key, val)  # Log the provided value
        else:
            v = self.episode_dict[key]  # Retrieve stored values for the key
            vals = np.concatenate(v) if isinstance(
                v[0], np.ndarray) and len(v[0].shape) > 0 else v  # Flatten values if they are arrays
            stats = mpi_statistics_scalar(
                vals, with_min_and_max=with_min_and_max)  # Calculate statistics
            super().log_tabular(
                key if average_only else 'Average' + key, stats[0])  # Log average
            if not (average_only):
                super().log_tabular('Std'+key, stats[1])  # Log standard deviation
            if with_min_and_max:
                super().log_tabular('Max'+key, stats[3])  # Log maximum value
                super().log_tabular('Min'+key, stats[2])  # Log minimum value
        self.episode_dict[key] = []  # Clear stored values for the key

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.episode_dict[key]  # Retrieve stored values for the key
        vals = np.concatenate(v) if isinstance(
            v[0], np.ndarray) and len(v[0].shape) > 0 else v  # Flatten values if they are arrays
        return mpi_statistics_scalar(vals)  # Return computed statistics
