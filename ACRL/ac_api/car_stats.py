import os  # Import the os module for operating system dependent functionality
import sys  # Import the sys module for system-specific parameters and functions
import platform  # Import the platform module to access underlying platform's identifying data

APP_NAME = 'ACRL'  # Define the application name

# Attempt to add third-party libraries to the system path
try:
    # Check if the system architecture is 64-bit or not
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Set the directory for 64-bit libraries
    else:
        sysdir = "stdlib"  # Set the directory for 32-bit libraries
    
    # Insert the path for the application's third-party libraries into sys.path
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME)
    )
    
    # Append the current directory to the system PATH environment variable
    os.environ['PATH'] += ";."
    
    # Insert the path for the appropriate standard library directory into sys.path
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    
    # Again append the current directory to the system PATH environment variable
    os.environ['PATH'] += ";."
except Exception as e:
    # Log an error message if there is an issue importing libraries
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Importing necessary modules from the ACRL application
import ac  # noqa: E402  # Import the ac module (specific to the ACRL application)
import acsys  # noqa: E402  # Import the acsys module (specific to the ACRL application)
from sim_info import info  # noqa: E402  # Import the info object from the sim_info module

def get_has_drs() -> int:
    """
    Retrieves whether the car driven by the player has DRS (Drag Reduction System)
    :return: 0 if no DRS, 1 if there is DRS
    """
    return info.static.hasDRS  # Return the DRS status from the info object

def get_has_ers() -> int:
    """
    Retrieves whether the car driven by the player has ERS (Energy Recovery System)
    :return: 0 if no ERS, 1 if there is ERS
    """
    return info.static.hasERS  # Return the ERS status from the info object

def get_has_kers() -> int:
    """
    Retrieves whether the car driven by the player has KERS (Kinetic Energy Recovery System)
    :return: 0 if no KERS, 1 if there is KERS
    """
    return info.static.hasKERS  # Return the KERS status from the info object

def abs_level() -> int:
    """
    Retrieves the ABS (Anti-lock Braking System) level active for the car driven by the player
    Note: This function seems to be buggy
    :return: value between 0 and 1, the higher the value, the stronger the ABS
    """
    return info.physics.abs  # Return the ABS level from the physics information

def get_max_rpm() -> int:
    """
    Retrieves the maximum RPM (Revolutions Per Minute) of the car driven by the player
    :return: the maximum RPM, defaulting to 1000000 if not set
    """
    if info.static.maxRpm:
        return info.static.maxRpm  # Return the maximum RPM if available
    else:
        return 1000000  # Return a default value if maximum RPM is not set

def get_max_fuel() -> int:
    """
    Retrieves the maximum fuel capacity of the car driven by the player
    :return: the maximum fuel (in KG, or possibly in Liters)
    """
    return info.static.maxFuel  # Return the maximum fuel capacity from the static information
