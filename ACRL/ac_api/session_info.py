import os  # Import the os module for interacting with the operating system
import sys  # Import the sys module for system-specific parameters and functions
import platform  # Import the platform module to access underlying platform details

APP_NAME = 'ACRL'  # Define the application name as a constant

# Add the third party libraries to the path
try:
    # Check if the system architecture is 64-bit or 32-bit
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Set the directory for 64-bit standard libraries
    else:
        sysdir = "stdlib"  # Set the directory for 32-bit standard libraries

    # Insert the path to the third-party libraries for the application into sys.path
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    os.environ['PATH'] += ";."  # Append the current directory to the system PATH
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))  # Add the standard library path
    os.environ['PATH'] += ";."  # Append the current directory to the system PATH again
except Exception as e:
    # Log any exceptions that occur during the import process
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Import necessary modules from the Assetto Corsa API
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402

# Function to retrieve the current session type
# Returns type ACC_SESSION_TYPE
# Note: This function only works after a certain number of ticks after loading into the game
def get_session_type():
    """
    Retrieve session type. 
    unknown = -1, practice = 0, qualifying = 1, race = 2, hotlap = 3, timeattack = 4, rest irrelevant
    :return: type ACC_SESSION_TYPE current session type
    """
    return info.graphics.session  # Return the current session type

def get_driver_name() -> str:
    """
    Retrieve nickname of the driver of a car
    :return: driver name
    """
    return info.static.playerNick  # Return the player's nickname

def get_car_name(car: int = 0) -> str:
    """
    Retrieve name of a car. Car type (e.g. La Ferrari)
    :param car: the car selected (user is 0)
    :return: car name
    """
    return ac.getCarName(car)  # Return the name of the specified car

def get_track_name() -> str:
    """
    Retrieve name of the current track driven
    :return: track driven
    """
    return ac.getTrackName(0)  # Return the name of the current track

def get_track_config() -> str:
    """
    Retrieve configuration of track driven
    :return: configuration of track
    """
    return ac.getTrackConfiguration(0)  # Return the configuration of the current track

def get_track_length() -> float:
    """
    Retrieve the track length
    :return: track length in meters
    """
    return ac.getTrackLength(0)  # Return the length of the current track in meters

def get_cars_count() -> int:
    """
    Retrieve session's max number of cars
    :return: maximum car count in current session
    """
    return ac.getCarsCount()  # Return the maximum number of cars in the current session

def get_session_status() -> int:
    """
    Retrieve the status of current session 
    0=OFF, 1=REPLAY, 2=LIVE, 3=PAUSE
    :return: session status
    """
    return info.graphics.status  # Return the current session status

# Function to get the ballast weight of a car (in kg)
def get_car_ballast(car: int = 0) -> int:
    return ac.getCarBallast(car)  # Return the ballast weight of the specified car

# Function to get the caster angle of a car (in radians)
def get_caster(car: int = 0):
    return ac.getCarState(car, acsys.CS.Caster)  # Return the caster angle of the specified car

# Function to get the radius of each tyre (returns only one tyre's radius)
def get_radius(car: int = 0):
    return ac.getCarState(car, acsys.CS.TyreRadius)[0]  # Return the radius of the first tyre

def get_car_min_height(car: int = 0) -> int:
    return ac.getCarMinHeight(car)  # Return the minimum height of the specified car

def get_car_ffb() -> int:
    return ac.getCarFFB()  # Return the force feedback setting of the car

def get_air_temp():
    return info.physics.airTemp  # Return the current air temperature

def get_air_density():
    return info.physics.airDensity  # Return the current air density

def get_road_temp():
    return info.physics.roadTemp  # Return the current road temperature

def get_tyre_compound():
    return info.graphics.tyreCompound  # Return the current tyre compound in use

# Function to get the surface grip (between 0% and 100%)
def get_surface_grip():
    return info.graphics.surfaceGrip  # Return the current surface grip percentage

def get_max_torque():
    return info.static.maxTorque  # Return the maximum torque of the car

def get_max_power():
    return info.static.maxPower  # Return the maximum power output of the car

def get_max_rpm():
    return info.static.maxRpm  # Return the maximum RPM of the car

def get_max_sus_travel():
    return info.static.suspensionMaxTravel  # Return the maximum suspension travel of the car

def get_max_turbo():
    return info.static.maxTurboBoost  # Return the maximum turbo boost of the car

# Function to get all the assists available in the game
def get_assists():
    fuel = info.static.aidFuelRate  # Retrieve fuel rate assist status
    line = info.graphics.idealLineOn  # Retrieve ideal line assist status
    tyre = info.static.aidTireRate  # Retrieve tyre wear assist status
    dmg = info.static.aidMechanicalDamage  # Retrieve mechanical damage assist status
    blankets = info.static.aidAllowTyreBlankets  # Retrieve tyre blankets assist status
    stability = info.static.aidStability  # Retrieve stability assist status
    clutch = info.static.aidAutoClutch  # Retrieve auto clutch assist status
    blip = info.static.aidAutoBlip  # Retrieve auto blip assist status
    
    # Return a tuple containing the status of all assists
    res = (fuel, line, tyre, dmg, blankets, stability, clutch, blip)
    return res
