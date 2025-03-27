import os
import sys
import platform

APP_NAME = 'ACRL'  # Define the application name

# Add the third-party libraries to the Python path
try:
    # Determine the system architecture (64-bit or 32-bit)
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Use 64-bit library directory
    else:
        sysdir = "stdlib"  # Use 32-bit library directory

    # Insert the path for third-party libraries for the application
    sys.path.insert(
        len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME)
    )
    
    # Append the current directory to the system PATH
    os.environ['PATH'] += ";."
    
    # Insert the appropriate system directory for third-party libraries
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    os.environ['PATH'] += ";."  # Append current directory again
except Exception as e:
    # Log any errors encountered during the import process
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Import necessary modules from the AC library and simulation info
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402


def format_time(millis: int) -> str:
    """
    Formats the time from timestamp to readable time
    :param millis: timestamp in milliseconds
    :return: formatted string in format mm:ss:ms
    """
    m = int(millis / 60000)  # Calculate minutes
    s = int((millis % 60000) / 1000)  # Calculate seconds
    ms = millis % 1000  # Calculate milliseconds

    return "{:02d}:{:02d}.{:03d}".format(m, s, ms)  # Return formatted time


def get_speed(car: int = 0, unit: str = "kmh") -> float:
    """
    Retrieve the current speed of a car
    :param car: the car selected (user is 0)
    :param unit: either kmh or mph or ms on how to show speed
    :return: current speed [0, ...]
    """
    # Return speed based on the requested unit
    if unit == "kmh":
        return ac.getCarState(car, acsys.CS.SpeedKMH)
    elif unit == "mph":
        return ac.getCarState(car, acsys.CS.SpeedMPH)
    elif unit == "ms":
        return ac.getCarState(car, acsys.CS.SpeedMS)


def get_delta_to_car_ahead(formatted: bool = False):
    """
    Retrieve time delta to the car ahead
    :param formatted: true if format should be in readable str
    :return: delta to car ahead in calculated time distance (float) or string format
    """
    import ac_api.session_info as si  # Import session information
    import ac_api.lap_info as li  # Import lap information
    time = 0  # Initialize time delta
    dist = 0  # Initialize distance delta
    track_len = si.get_track_length()  # Get the track length
    lap = li.get_lap_count(0)  # Get the player's lap count
    pos = get_location(0)  # Get the player's position on the track

    # Loop through all cars to find the one directly ahead
    for car in range(si.get_cars_count()):
        if get_position(car) == get_position(0) - 1:  # Check if this car is ahead
            lap_next = li.get_lap_count(car)  # Get the lap count of the car ahead
            pos_next = get_location(car)  # Get the position of the car ahead

            # Calculate distance to the car ahead
            dist = max(0, (pos_next * track_len + lap_next *
                           track_len) - (pos * track_len + lap * track_len))
            # Calculate time delta based on distance and speed
            time = max(0.0, dist / max(10.0, get_speed(0, "ms")))
            break  # Exit loop after finding the car ahead

    # Return time delta in the requested format
    if not formatted:
        return time
    else:
        if dist > track_len:  # If the distance exceeds the track length
            laps = dist / track_len  # Calculate laps
            if laps > 1:
                return "+{:3.1f}".format(laps) + " Laps"  # Return in laps
            else:
                return "+{:3.1f}".format(laps) + " Lap"  # Return in laps
        elif time > 60:  # If time exceeds 60 seconds
            return "+" + format_time(int(time * 1000))  # Return formatted time
        else:
            return "+{:3.3f}".format(time)  # Return time in seconds


def get_delta_to_car_behind(formatted: bool = False):
    """
    Retrieve time delta to the car behind
    :param formatted: true if format should be in readable str
    :return: delta to car behind in calculated time distance (float) or string format
    """
    import ac_api.session_info as si  # Import session information
    import ac_api.lap_info as li  # Import lap information
    time = 0  # Initialize time delta
    dist = 0  # Initialize distance delta
    track_len = si.get_track_length()  # Get the track length
    lap = li.get_lap_count(0)  # Get the player's lap count
    pos = get_location(0)  # Get the player's position on the track

    # Loop through all cars to find the one directly behind
    for car in range(si.get_cars_count()):
        if get_position(car) == get_position(0) + 1:  # Check if this car is behind
            lap_next = li.get_lap_count(car)  # Get the lap count of the car behind
            pos_next = get_location(car)  # Get the position of the car behind

            # Calculate distance to the car behind
            dist = max(0, (pos * track_len + lap * track_len) -
                       (pos_next * track_len + lap_next * track_len))
            # Calculate time delta based on distance and speed
            time = max(0.0, dist / max(10.0, get_speed(car, "ms")))
            break  # Exit loop after finding the car behind

    # Return time delta in the requested format
    if not formatted:
        return time
    else:
        if dist > track_len:  # If the distance exceeds the track length
            laps = dist / track_len  # Calculate laps
            if laps > 1:
                return "-{:3.1f}".format(laps) + " Laps"  # Return in laps
            else:
                return "-{:3.1f}".format(laps) + " Lap"  # Return in laps
        elif time > 60:  # If time exceeds 60 seconds
            return "-" + format_time(int(time * 1000))  # Return formatted time
        else:
            return "-{:3.3f}".format(time)  # Return time in seconds


def get_location(car: int = 0) -> float:
    """
    Retrieve current location of a car
    :param car: the car selected (user is 0)
    :return: position on track relative with the lap between 0 and 1
    """
    return ac.getCarState(car, acsys.CS.NormalizedSplinePosition)  # Get normalized position


def get_world_location(car: int = 0):
    """
    Retrieve absolute location of a car
    :param car: the car selected (user is 0)
    :return: absolute location [x,y,z] ((0,x,0) is the middle)
    """
    x = ac.getCarState(car, acsys.CS.WorldPosition)[0]  # Get x-coordinate
    y = ac.getCarState(car, acsys.CS.WorldPosition)[1]  # Get y-coordinate
    z = ac.getCarState(car, acsys.CS.WorldPosition)[2]  # Get z-coordinate
    res = (x, y, z)  # Create a tuple for the coordinates
    return res  # Return the coordinates


def get_position(car: int = 0) -> int:
    """
    Retrieve current driving position of a car
    :param car: the car selected (user is 0)
    :return: position of car (0 is the lead car)
    """
    return ac.getCarRealTimeLeaderboardPosition(car) + 1  # Get the leaderboard position


def get_drs_available():
    """
    Check if DRS is available for the player's car
    :return: DRS availability status (0 if disabled, 1 if enabled)
    """
    return info.physics.drsAvailable


def get_drs_enabled() -> bool:
    """
    Check whether DRS of the car of the player is enabled
    :return: DRS enabled status
    """
    return info.physics.drsEnabled  # Return DRS enabled status


def get_gear(car: int = 0, formatted: bool = True):
    """
    Retrieve current gear of a car. If formatted, it returns string, if not, it returns int.
    :param car: the car selected (user is 0)
    :param formatted: boolean to format result or not.
    :return: current gear of car as integer or string format
    """
    gear = ac.getCarState(car, acsys.CS.Gear)  # Get the current gear
    if formatted:
        if gear == 0:
            return "R"  # Reverse
        elif gear == 1:
            return "N"  # Neutral
        else:
            return str(gear - 1)  # Return gear as string (0-indexed)
    else:
        return gear  # Return gear as integer


def get_rpm(car: int = 0) -> float:
    """
    Retrieve rpm of a car
    :param car: the car selected (user is 0)
    :return: rpm of a car [0, ...]
    """
    return ac.getCarState(car, acsys.CS.RPM)  # Get RPM


def get_fuel() -> float:
    """
    Retrieve amount of fuel in player's car in kg
    :return: amount of fuel [0, ...]
    """
    return info.physics.fuel  # Get fuel amount


def get_tyres_off_track() -> int:
    """
    Retrieve amount of tyres of player's car off-track
    :return: amount of tyres off-track [0,4]
    """
    return info.physics.numberOfTyresOut  # Get number of tyres off track


def get_car_in_pit_lane() -> bool:
    """
    Retrieve whether player's car is in the pitlane
    :return: car in pit lane status
    """
    return info.graphics.isInPitLane  # Check if in pit lane


def get_location_damage(loc: str = "front") -> float:
    """
    Retrieve car damage per side
    :param loc: front, rear, left or right
    :return: damage [0, ...]
    """
    if loc == "front":
        return info.physics.carDamage[0]  # Front damage
    elif loc == "rear":
        return info.physics.carDamage[1]  # Rear damage
    elif loc == "left":
        return info.physics.carDamage[2]  # Left damage
    elif loc == "right":
        return info.physics.carDamage[3]  # Right damage
    else:
        # Centre damage
        return info.physics.carDamage[4]


def get_total_damage():
    """
    Retrieve total damage across all sides of the car
    :return: tuple of damage values (front, rear, left, right, centre)
    """
    front = info.physics.carDamage[0]  # Front damage
    rear = info.physics.carDamage[1]  # Rear damage
    left = info.physics.carDamage[2]  # Left damage
    right = info.physics.carDamage[3]  # Right damage
    centre = info.physics.carDamage[4]  # Centre damage
    res = (front, rear, left, right, centre)  # Create a tuple of damages
    return res  # Return the tuple


def get_cg_height(car: int = 0) -> float:
    """
    Retrieve height of the center of gravity of the car from the ground
    :param car: the car selected (user is 0)
    :return: height of the center of gravity [0, ...]
    """
    return ac.getCarState(car, acsys.CS.CGHeight)  # Get centre of gravity height


def get_drive_train_speed(car: int = 0):
    """
    Retrieve speed delivered to the wheels
    :param car: the car selected (user is 0)
    :return: speed delivered to the wheels [0, ...]
    """
    return ac.getCarState(car, acsys.CS.DriveTrainSpeed)  # Get drive train speed


def get_velocity():
    """
    Retrieve velocity of the car in 3D coordinates
    :return: tuple of velocity in coordinates (x, y, z)
    """
    x = info.physics.velocity[0]  # Get x velocity
    y = info.physics.velocity[1]  # Get y velocity
    z = info.physics.velocity[2]  # Get z velocity
    res = (x, y, z)  # Create a tuple for the velocity
    return res  # Return the velocity


def get_acceleration():
    """
    Retrieve acceleration of the car in 3D coordinates
    :return: tuple of acceleration in coordinates (x, y, z)
    """
    x = info.physics.accG[0]  # Get x acceleration
    y = info.physics.accG[1]  # Get y acceleration
    z = info.physics.accG[2]  # Get z acceleration
    res = (x, y, z)  # Create a tuple for the acceleration
    return res  # Return the acceleration


def get_tc_in_action():
    """
    Check if traction control is active
    :return: traction control status
    """
    return info.physics.tc  # Return traction control status


def get_abs_in_action():
    """
    Check if anti-lock braking system is active
    :return: ABS status
    """
    return info.physics.abs  # Return ABS status


def get_brake_bias():
    """
    Retrieve front brake bias percentage
    :return: brake bias [0, 1]
    """
    return info.physics.brakeBias  # Get brake bias


def get_engine_brake():
    """
    Retrieve engine brake settings
    :return: engine brake settings
    """
    return info.physics.engineBrake  # Get engine brake settings
