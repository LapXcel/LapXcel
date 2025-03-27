import os  # Module for interacting with the operating system
import sys  # Module for system-specific parameters and functions
import platform  # Module to access underlying platform's identifying data

APP_NAME = 'ACRL'  # Define the application name

# Add the third-party libraries to the Python path
try:
    # Check the architecture of the operating system (64-bit or 32-bit)
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Set directory for 64-bit libraries
    else:
        sysdir = "stdlib"  # Set directory for 32-bit libraries

    # Insert the path for third-party libraries into sys.path
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    os.environ['PATH'] += ";."  # Append current directory to PATH

    # Insert the appropriate standard library path into sys.path
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    os.environ['PATH'] += ";."  # Append current directory to PATH again
except Exception as e:
    # Log any errors that occur while importing libraries
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Import necessary modules from the AC (Assetto Corsa) API
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402

"""
Constants for tyre indices:
0 = Front Left (FL), 1 = Front Right (FR), 2 = Rear Left (RL), 3 = Rear Right (RR)
"""

def get_tyre_wear_value(tyre: int) -> float:
    """
    Retrieve tyre wear of a specified tyre.
    100 indicates mint condition, 0 indicates fully worn (puncture).
    :param tyre: int [0,3] - index of the tyre
    :return: float - tyre wear percentage [0,100]
    """
    return info.physics.tyreWear[tyre]  # Access tyre wear data from simulation info

def get_tyre_dirty(tyre: int) -> float:
    """
    Retrieve the "dirty level" of a specified tyre.
    0 indicates clean, 5 indicates most dirty.
    :param tyre: int [0,3] - index of the tyre
    :return: float - dirty level [0,5]
    """
    return info.physics.tyreDirtyLevel[tyre]  # Access tyre dirty level data

def get_tyre_temp(tyre: int, loc: str) -> float:
    """
    Retrieve the temperature of a tyre at a specified location.
    :param tyre: int [0,3] - index of the tyre
    :param loc: str - location ('i' for inner, 'm' for middle, 'o' for outer, 'c' for core)
    :return: float - temperature of the tyre at the specified location
    """
    # Return temperature based on specified location
    if loc == "i":
        return info.physics.tyreTempI[tyre]  # Inner temperature
    elif loc == "m":
        return info.physics.tyreTempM[tyre]  # Middle temperature
    elif loc == "o":
        return info.physics.tyreTempO[tyre]  # Outer temperature
    elif loc == "c":
        return info.physics.tyreCoreTemperature[tyre]  # Core temperature

def get_tyre_pressure(tyre: int) -> float:
    """
    Retrieve the pressure of a specified tyre.
    :param tyre: int [0,3] - index of the tyre
    :return: float - tyre pressure
    """
    return info.physics.wheelsPressure[tyre]  # Access tyre pressure data

# Stays 26.0 for some reason
def get_brake_temp(loc: int = 0) -> float:
    """
    Retrieve the temperature of a brake at a specified location.
    :param loc: int [0,3] - index of the brake
    :return: float - brake temperature
    """
    return info.physics.brakeTemp[loc]  # Access brake temperature data

# Functions below return a 4D vector

def get_slip_ratio(car: int = 0):
    """
    Retrieve the slip ratio of the specified car.
    Slip ratio is a value between 0 and 1.
    :param car: int - index of the car
    :return: float - slip ratio
    """
    return ac.getCarState(car, acsys.CS.SlipRatio)  # Get slip ratio from car state

def get_slip_angle(car: int = 0):
    """
    Retrieve the slip angle of the specified car.
    Slip angle is the angle between the desired direction and the actual direction of the vehicle.
    :param car: int - index of the car
    :return: float - slip angle in degrees [0, 360]
    """
    return ac.getCarState(car, acsys.CS.SlipAngle)  # Get slip angle from car state

def get_camber(car: int = 0):
    """
    Retrieve the camber angle of the specified car.
    :param car: int - index of the car
    :return: float - camber angle in degrees
    """
    return ac.getCarState(car, acsys.CS.CamberDeg)  # Get camber angle from car state

def get_torque(car: int = 0):
    """
    Retrieve the self-aligning torque of the specified car.
    :param car: int - index of the car
    :return: float - self-aligning torque value
    """
    return ac.getCarState(car, acsys.CS.Mz)  # Get torque from car state

def get_load(car: int = 0):
    """
    Retrieve the load on each tyre of the specified car.
    :param car: int - index of the car
    :return: float - load value on tyres
    """
    return ac.getCarState(car, acsys.CS.Load)  # Get load from car state

def get_suspension_travel(car: int = 0):
    """
    Retrieve the vertical suspension travel of the specified car.
    :param car: int - index of the car
    :return: float - vertical suspension travel value
    """
    return ac.getCarState(car, acsys.CS.SuspensionTravel)  # Get suspension travel from car state

def get_tyre_contact_normal(car: int = 0, tyre: int = 0):
    """
    Retrieve the normal vector to the tyre's contact point.
    :param car: int - index of the car
    :param tyre: int - index of the tyre
    :return: tuple - normal vector (x, y, z)
    """
    return ac.getCarState(car, acsys.CS.TyreContactNormal, tyre)  # Get contact normal vector

def get_tyre_contact_point(car: int = 0, tyre: int = 0):
    """
    Retrieve the contact point of the tyre with the tarmac.
    :param car: int - index of the car
    :param tyre: int - index of the tyre
    :return: tuple - contact point (x, y, z)
    """
    return ac.getCarState(car, acsys.CS.TyreContactPoint, tyre)  # Get contact point coordinates

def get_tyre_heading_vector(tyre: int = 0):
    """
    Retrieve the heading vector of the tyre.
    :param tyre: int - index of the tyre
    :return: tuple - heading vector (x, y, z)
    """
    x = info.physics.tyreContactHeading[0][tyre]  # X component of heading vector
    y = info.physics.tyreContactHeading[1][tyre]  # Y component of heading vector
    z = info.physics.tyreContactHeading[2][tyre]  # Z component of heading vector
    res = (x, y, z)  # Combine components into a tuple
    return res  # Return heading vector

def get_tyre_right_vector(car: int = 0, tyre: int = 0):
    """
    Retrieve the right vector of the tyre.
    :param car: int - index of the car
    :param tyre: int - index of the tyre
    :return: tuple - right vector
    """
    return ac.getCarState(car, acsys.CS.TyreRightVector, tyre)  # Get right vector of tyre

def get_angular_speed(tyre: int = 0):
    """
    Retrieve the angular speed of a specified tyre.
    :param tyre: int - index of the tyre
    :return: float - angular speed in rad/s
    """
    return info.physics.wheelAngularSpeed[tyre]  # Access wheel angular speed data
