import os  # Import the os module for interacting with the operating system

from setuptools import find_packages, setup  # Import necessary functions from setuptools for packaging

# Read the version number from the version.txt file located in the 'crossq' directory
with open(os.path.join("crossq", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()  # Store the version number as a string

# Setup function to configure the package
setup(
    name="crossq-rl",  # Name of the package
    packages=[package for package in find_packages() if package.startswith("crossq")],  # Automatically find packages starting with 'crossq'
    package_data={"cross": ["py.typed", "version.txt"]},  # Include additional package data files
    install_requires=[  # List of dependencies required to install the package
        "stable_baselines3==2.1.0",  # Reinforcement learning library
        "flax==0.7.4",  # Neural network library for JAX
        "gymnasium==0.29.1",  # Toolkit for developing and comparing reinforcement learning algorithms
        "imageio==2.31.3",  # Library for reading and writing images
        "mujoco==2.3.7",  # Physics engine for simulating rigid body dynamics
        "optax==0.1.7",  # Gradient processing and optimization library for JAX
        "tqdm==4.66.1",  # Library for progress bars
        "rich==13.5.2",  # Library for rich text and beautiful formatting in the terminal
        "rlax==0.1.6",  # Reinforcement learning library built on JAX
        "tensorboard==2.14.0",  # TensorBoard for visualizing training runs
        "tensorflow-probability==0.21.0",  # Probabilistic reasoning and statistical analysis library
        "wandb==0.15.10",  # Tool for experiment tracking and visualization
        "scipy==1.11.4",  # Library for scientific computing
        "shimmy==1.3.0"  # Library for building reinforcement learning environments
    ],
    extras_require={  # Optional dependencies for additional functionality
        "tests": [  # Dependencies required for testing
            "pytest",  # Testing framework
            "pytest-cov",  # Coverage reporting for tests
            "pytest-env",  # Set environment variables for tests
            "pytest-xdist",  # Run tests in parallel
            # Type check
            "mypy",  # Static type checker for Python
            # Lint code
            "ruff",  # Linter for Python code
            # Sort imports
            "isort>=5.0",  # Tool for sorting imports
            # Reformat
            "black",  # Code formatter
        ],
    },
    description="Jax version of CrossQ; Bhatt and Palenicek et al. 2023.",  # Short description of the package
    author="Aditya Bhatt, Daniel Palenicek",  # Authors of the package
    url="https://github.com/adityab/sbx-crossq",  # URL for the package's repository
    author_email="aditya.bhatt@dfki.de, daniel.palenicek@tu-darmstadt.de",  # Authors' email addresses
    keywords="crossq reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",  # Keywords for the package
    license="MIT",  # License type for the package
    version=__version__,  # Version of the package read from version.txt
    python_requires="==3.11.5",  # Specify the required Python version
    # Classifiers for the package, useful for PyPI
    classifiers=[
        "Programming Language :: Python :: 3.11",  # Specify the programming language and version
    ],
)
