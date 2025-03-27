import os  # Import the os module for interacting with the operating system
import sys  # Import the sys module for manipulating the Python runtime environment
import platform  # Import the platform module to access information about the system

APP_NAME = 'ACRL'  # Define the application name

# Add the third-party libraries to the Python path
try:
    # Check if the system architecture is 64-bit or 32-bit
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Set the directory for 64-bit libraries
    else:
        sysdir = "stdlib"  # Set the directory for 32-bit libraries

    # Insert the path for the third-party libraries into sys.path
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    # Append the current directory to the system PATH
    os.environ['PATH'] += ";."
    # Insert the appropriate third-party library directory into sys.path
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    # Append the current directory to the system PATH again
    os.environ['PATH'] += ";."
except Exception as e:
    # Log any errors that occur during the import of libraries
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Import the 'ac' module for accessing the Assetto Corsa API
import ac  # noqa: E402
# Import the 'acsys' module for accessing Assetto Corsa system constants
import acsys  # noqa: E402


def get_gas_input(car: int = 0) -> float:
    """
    Retrieve the gas input given to a car.
    :param car: the car selected (user is 0)
    :return: gas input between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.Gas)  # Return the gas input state of the specified car


def get_brake_input(car: int = 0) -> float:
    """
    Retrieve the brake input given to a car.
    :param car: the car selected (user is 0)
    :return: brake input between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.Brake)  # Return the brake input state of the specified car


def get_clutch(car: int = 0) -> float:
    """
    Retrieve the clutch status in the game of a car.
    :param car: the car selected (user is 0)
    :return: deployment of the clutch (1 is fully deployed, 0 is not deployed).
    """
    return ac.getCarState(car, acsys.CS.Clutch)  # Return the clutch state of the specified car


def get_steer_input(car: int = 0) -> float:
    """
    Retrieve the steering input given to a car.
    :param car: the car selected (user is 0)
    :return: steering input to the car, depends on the settings in AC, in degrees
    """
    return ac.getCarState(car, acsys.CS.Steer)  # Return the steering input state of the specified car

# Function to retrieve the last force feedback value for a car
def get_last_ff(car: int = 0) -> float:
    return ac.getCarState(car, acsys.CS.LastFF)  # Return the last force feedback state of the specified car
