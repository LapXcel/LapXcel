import os  # Importing the os module for operating system dependent functionality
import sys  # Importing the sys module for system-specific parameters and functions
import platform  # Importing the platform module to access underlying platform data

APP_NAME = 'ACRL'  # Define the application name

# Add the third-party libraries to the path
try:
    # Check the architecture of the platform (64-bit or 32-bit)
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"  # Use 'stdlib64' for 64-bit systems
    else:
        sysdir = "stdlib"  # Use 'stdlib' for 32-bit systems

    # Insert the path to the third-party libraries specific to the application
    sys.path.insert(len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    
    # Update the PATH environment variable to include the current directory
    os.environ['PATH'] += ";."
    
    # Insert the path to the specific standard library directory based on architecture
    sys.path.insert(len(sys.path), os.path.join(
        'apps/python/{}/third_party'.format(APP_NAME), sysdir))
    
    # Update the PATH environment variable again to include the current directory
    os.environ['PATH'] += ";."
except Exception as e:
    # Log an error message if there's an issue importing libraries
    ac.log("[ACRL] Error importing libraries: %s" % e)

# Importing necessary modules from the AC racing simulation API
import ac  # noqa: E402
import acsys  # noqa: E402
from sim_info import info  # noqa: E402


def format_time(millis: int) -> str:
    """
    Format time takes an integer representing milliseconds and turns it into a readable string.
    :param millis: the amount of milliseconds
    :return: formatted string [minutes, seconds, milliseconds]
    """
    # Convert milliseconds to minutes, seconds, and remaining milliseconds
    m = int(millis / 60000)
    s = int((millis % 60000) / 1000)
    ms = millis % 1000

    # Return the formatted time as a string
    return "{:02d}:{:02d}.{:03d}".format(m, s, ms)


def get_current_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieves the current lap time of the selected car.
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable string
    :return: current lap time in milliseconds (int) or string format
    """
    if formatted:
        # Get the current lap time for the specified car
        time = ac.getCarState(car, acsys.CS.LapTime)
        if time > 0:
            return format_time(time)  # Return formatted time if valid
        else:
            return "--:--"  # Return placeholder for invalid time
    else:
        return ac.getCarState(car, acsys.CS.LapTime)  # Return raw lap time


def get_last_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieves the last lap time of the selected car.
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable string
    :return: last lap time in milliseconds (int) or string format
    """
    if formatted:
        # Get the last lap time for the specified car
        time = ac.getCarState(car, acsys.CS.LastLap)
        if time > 0:
            return format_time(time)  # Return formatted time if valid
        else:
            return "--:--"  # Return placeholder for invalid time
    else:
        return ac.getCarState(car, acsys.CS.LastLap)  # Return raw last lap time


def get_best_lap_time(car: int = 0, formatted: bool = False):
    """
    Retrieve the best lap time recorded, does not save if invalidated lap.
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable string
    :return: best lap time in string format or formatted string
    """
    if formatted:
        # Get the best lap time for the specified car
        time = ac.getCarState(car, acsys.CS.BestLap)
        if time > 0:
            return format_time(time)  # Return formatted time if valid
        else:
            return "--:--"  # Return placeholder for invalid time
    else:
        return ac.getCarState(car, acsys.CS.BestLap)  # Return raw best lap time


def get_splits(car: int = 0, formatted: bool = False):
    """
    Retrieve the split times of the completed lap.
    :param car: the car selected (user is 0)
    :param formatted: true if format should be in readable string
    :return: list containing the splits in milliseconds (int) or string format
    """
    if formatted:
        # Get the last split times for the specified car
        times = ac.getLastSplits(car)
        formattedtimes = []

        if len(times) != 0:
            # Format each split time and append to the list
            for t in times:
                formattedtimes.append(format_time(t))
            return formattedtimes  # Return the list of formatted split times
        else:
            return "--:--"  # Return placeholder for no splits
    else:
        return ac.getLastSplits(car)  # Return raw split times


def get_split() -> str:
    """
    Retrieve the last sector split, but will return nothing if the last sector is the completion of a lap.
    :return: split in string format
    """
    return info.graphics.split  # Return the current split information


def get_invalid(car: int = 0) -> bool:
    """
    Retrieve if the current lap is invalid.
    :param car: the car selected (user is 0)
    :return: Invalid lap in boolean form
    """
    import ac_api.car_info as ci  # Importing car info module for additional checks

    # Check if the lap is invalidated or if the car has gone off track
    return ac.getCarState(car, acsys.CS.LapInvalidated) or ci.get_tyres_off_track() > 2


def get_lap_count(car: int = 0) -> int:
    """
    Retrieve the current number of laps.
    :param car: the car selected (user is 0)
    :return: The current number of laps (added by 1 default)
    """
    # Return the current lap count, incremented by 1 to account for zero-based indexing
    return ac.getCarState(car, acsys.CS.LapCount) + 1


def get_laps() -> str:
    """
    Returns the total number of laps in a race (only in a race).
    :return: total number of race laps
    """
    if info.graphics.numberOfLaps > 0:
        return info.graphics.numberOfLaps  # Return total laps if available
    else:
        return "-"  # Return placeholder if not in a race


def get_lap_delta(car: int = 0) -> float:
    """
    Retrieves the delta to the fastest lap.
    :param car: the car selected (user is 0)
    :return: delta to the fastest lap in seconds (float)
    """
    return ac.getCarState(car, acsys.CS.PerformanceMeter)  # Return the performance meter value


# Function to get the current sector index of the car
def get_current_sector():
    return info.graphics.currentSectorIndex  # Return the current sector index
