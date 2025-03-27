import mmap
import functools
import ctypes
from ctypes import c_int32, c_float, c_wchar

# Define constants for various statuses and types used in the simulation
AC_STATUS = c_int32
AC_OFF = 0
AC_REPLAY = 1
AC_LIVE = 2
AC_PAUSE = 3

AC_SESSION_TYPE = c_int32
AC_UNKNOWN = -1
AC_PRACTICE = 0
AC_QUALIFY = 1
AC_RACE = 2
AC_HOTLAP = 3
AC_TIME_ATTACK = 4
AC_DRIFT = 5
AC_DRAG = 6

AC_FLAG_TYPE = c_int32
AC_NO_FLAG = 0
AC_BLUE_FLAG = 1
AC_YELLOW_FLAG = 2
AC_BLACK_FLAG = 3
AC_WHITE_FLAG = 4
AC_CHECKERED_FLAG = 5
AC_PENALTY_FLAG = 6

# Define a structure for the physics data of the simulation
class SPageFilePhysics(ctypes.Structure):
    _pack_ = 4  # Set packing alignment to 4 bytes
    _fields_ = [
        ('packetId', c_int32),  # Identifier for the packet
        ('gas', c_float),  # Gas pedal position
        ('brake', c_float),  # Brake pedal position
        ('fuel', c_float),  # Amount of fuel
        ('gear', c_int32),  # Current gear
        ('rpms', c_int32),  # Engine RPMs
        ('steerAngle', c_float),  # Steering angle
        ('speedKmh', c_float),  # Speed in kilometers per hour
        ('velocity', c_float * 3),  # Velocity vector (x, y, z)
        ('accG', c_float * 3),  # Acceleration in Gs (x, y, z)
        ('wheelSlip', c_float * 4),  # Slip ratio for each wheel
        ('wheelLoad', c_float * 4),  # Load on each wheel
        ('wheelsPressure', c_float * 4),  # Pressure for each wheel
        ('wheelAngularSpeed', c_float * 4),  # Angular speed for each wheel
        ('tyreWear', c_float * 4),  # Wear level for each tyre
        ('tyreDirtyLevel', c_float * 4),  # Dirt level for each tyre
        ('tyreCoreTemperature', c_float * 4),  # Core temperature for each tyre
        ('camberRAD', c_float * 4),  # Camber angle for each wheel
        ('suspensionTravel', c_float * 4),  # Suspension travel for each wheel
        ('drs', c_float),  # Drag Reduction System status
        ('tc', c_float),  # Traction control status
        ('heading', c_float),  # Heading angle
        ('pitch', c_float),  # Pitch angle
        ('roll', c_float),  # Roll angle
        ('cgHeight', c_float),  # Height of the center of gravity
        ('carDamage', c_float * 5),  # Damage levels for various parts of the car
        ('numberOfTyresOut', c_int32),  # Number of tyres out of the track
        ('pitLimiterOn', c_int32),  # Whether the pit limiter is active
        ('abs', c_float),  # Anti-lock braking system status
        ('kersCharge', c_float),  # KERS charge level
        ('kersInput', c_float),  # KERS input
        ('autoShifterOn', c_int32),  # Whether the auto shifter is active
        ('rideHeight', c_float * 2),  # Ride height for front and rear
        ('turboBoost', c_float),  # Turbo boost level
        ('ballast', c_float),  # Ballast weight
        ('airDensity', c_float),  # Air density
        ('airTemp', c_float),  # Air temperature
        ('roadTemp', c_float),  # Road temperature
        ('localAngularVel', c_float * 3),  # Local angular velocity (x, y, z)
        ('finalFF', c_float),  # Final force feedback
        ('performanceMeter', c_float),  # Performance meter value
        ('engineBrake', c_int32),  # Engine braking settings
        ('ersRecoveryLevel', c_int32),  # ERS recovery level
        ('ersPowerLevel', c_int32),  # ERS power level
        ('ersHeatCharging', c_int32),  # ERS heat charging status
        ('ersIsCharging', c_int32),  # Whether ERS is charging
        ('kersCurrentKJ', c_float),  # Current KERS energy in kilojoules
        ('drsAvailable', c_int32),  # Whether DRS is available
        ('drsEnabled', c_int32),  # Whether DRS is enabled
        ('brakeTemp', c_float * 4),  # Brake temperatures for each wheel
        ('clutch', c_float),  # Clutch position
        ('tyreTempI', c_float * 4),  # Inner tyre temperatures
        ('tyreTempM', c_float * 4),  # Middle tyre temperatures
        ('tyreTempO', c_float * 4),  # Outer tyre temperatures
        ('isAIControlled', c_int32),  # Whether the car is AI controlled
        ('tyreContactPoint', c_float * 4 * 3),  # Contact points for each tyre
        ('tyreContactNormal', c_float * 4 * 3),  # Contact normals for each tyre
        ('tyreContactHeading', c_float * 4 * 3),  # Contact headings for each tyre
        ('brakeBias', c_float),  # Brake bias setting
        ('localVelocity', c_float * 3),  # Local velocity vector (x, y, z)
    ]

# Define a structure for the graphical data of the simulation
class SPageFileGraphic(ctypes.Structure):
    _pack_ = 4  # Set packing alignment to 4 bytes
    _fields_ = [
        ('packetId', c_int32),  # Identifier for the packet
        ('status', AC_STATUS),  # Current status of the simulation
        ('session', AC_SESSION_TYPE),  # Current session type
        ('currentTime', c_wchar * 15),  # Formatted current time
        ('lastTime', c_wchar * 15),  # Formatted last lap time
        ('bestTime', c_wchar * 15),  # Formatted best lap time
        ('split', c_wchar * 15),  # Formatted split time
        ('completedLaps', c_int32),  # Number of completed laps
        ('position', c_int32),  # Current position in the race
        ('iCurrentTime', c_int32),  # Current time in milliseconds
        ('iLastTime', c_int32),  # Last lap time in milliseconds
        ('iBestTime', c_int32),  # Best lap time in milliseconds
        ('sessionTimeLeft', c_float),  # Time left in the current session
        ('distanceTraveled', c_float),  # Distance traveled in the current lap
        ('isInPit', c_int32),  # Whether the car is in the pit
        ('currentSectorIndex', c_int32),  # Index of the current sector
        ('lastSectorTime', c_int32),  # Last sector time in milliseconds
        ('numberOfLaps', c_int32),  # Total number of laps in the race
        ('tyreCompound', c_wchar * 33),  # Current tyre compound
        ('replayTimeMultiplier', c_float),  # Multiplier for replay speed
        ('normalizedCarPosition', c_float),  # Normalized position of the car on the track
        ('carCoordinates', c_float * 3),  # Car coordinates (x, y, z)
        ('penaltyTime', c_float),  # Current penalty time
        ('flag', AC_FLAG_TYPE),  # Current flag type
        ('idealLineOn', c_int32),  # Whether the ideal line is active
        ('isInPitLane', c_int32),  # Whether the car is in the pit lane
        ('surfaceGrip', c_float),  # Grip level of the surface
        ('mandatoryPitDone', c_int32),  # Whether the mandatory pit stop has been completed
        ('windSpeed', c_float),  # Current wind speed
        ('windDirection', c_float),  # Current wind direction
    ]

# Define a structure for static data that does not change during the simulation
class SPageFileStatic(ctypes.Structure):
    _pack_ = 4  # Set packing alignment to 4 bytes
    _fields_ = [
        ('_smVersion', c_wchar * 15),  # Version of the simulation
        ('_acVersion', c_wchar * 15),  # Version of the Assetto Corsa
        ('numberOfSessions', c_int32),  # Total number of sessions
        ('numCars', c_int32),  # Number of cars in the simulation
        ('carModel', c_wchar * 33),  # Car model name
        ('track', c_wchar * 33),  # Track name
        ('playerName', c_wchar * 33),  # Player's first name
        ('playerSurname', c_wchar * 33),  # Player's surname
        ('playerNick', c_wchar * 33),  # Player's nickname
        ('sectorCount', c_int32),  # Number of sectors on the track
        ('maxTorque', c_float),  # Maximum torque of the car
        ('maxPower', c_float),  # Maximum power of the car
        ('maxRpm', c_int32),  # Maximum RPM of the engine
        ('maxFuel', c_float),  # Maximum fuel capacity
        ('suspensionMaxTravel', c_float * 4),  # Maximum suspension travel for each wheel
        ('tyreRadius', c_float * 4),  # Radius of each tyre
        ('maxTurboBoost', c_float),  # Maximum turbo boost
        ('airTemp', c_float),  # Air temperature
        ('roadTemp', c_float),  # Road temperature
        ('penaltiesEnabled', c_int32),  # Whether penalties are enabled
        ('aidFuelRate', c_float),  # Fuel rate aid setting
        ('aidTireRate', c_float),  # Tyre wear aid setting
        ('aidMechanicalDamage', c_float),  # Mechanical damage aid setting
        ('aidAllowTyreBlankets', c_int32),  # Whether tyre blankets are allowed
        ('aidStability', c_float),  # Stability aid setting
        ('aidAutoClutch', c_int32),  # Auto clutch aid setting
        ('aidAutoBlip', c_int32),  # Auto blip aid setting
        ('hasDRS', c_int32),  # Whether the car has DRS
        ('hasERS', c_int32),  # Whether the car has ERS
        ('hasKERS', c_int32),  # Whether the car has KERS
        ('kersMaxJ', c_float),  # Maximum KERS energy in joules
        ('engineBrakeSettingsCount', c_int32),  # Number of engine brake settings
        ('ersPowerControllerCount', c_int32),  # Number of ERS power controllers
        ('trackSPlineLength', c_float),  # Length of the track spline
        ('trackConfiguration', c_wchar * 33),  # Track configuration
        ('ersMaxJ', c_float),  # Maximum ERS energy in joules
        ('isTimedRace', c_int32),  # Whether the race is timed
        ('hasExtraLap', c_int32),  # Whether there is an extra lap
        ('carSkin', c_wchar * 33),  # Skin of the car
        ('reversedGridPositions', c_int32),  # Whether grid positions are reversed
        ('pitWindowStart', c_int32),  # Start of the pit window
        ('pitWindowEnd', c_int32),  # End of the pit window
    ]

# Class to manage simulation information
class SimInfo:
    def __init__(self):
        # Memory-mapped files for physics, graphics, and static data
        self._acpmf_physics = mmap.mmap(0, ctypes.sizeof(SPageFilePhysics), "acpmf_physics")
        self._acpmf_graphics = mmap.mmap(0, ctypes.sizeof(SPageFileGraphic), "acpmf_graphics")
        self._acpmf_static = mmap.mmap(0, ctypes.sizeof(SPageFileStatic), "acpmf_static")
        
        # Create structures from the memory-mapped files
        self.physics = SPageFilePhysics.from_buffer(self._acpmf_physics)
        self.graphics = SPageFileGraphic.from_buffer(self._acpmf_graphics)
        self.static = SPageFileStatic.from_buffer(self._acpmf_static)

    def close(self):
        # Close the memory-mapped files
        self._acpmf_physics.close()
        self._acpmf_graphics.close()
        self._acpmf_static.close()

    def __del__(self):
        # Ensure resources are released when the object is deleted
        self.close()

# Create an instance of the SimInfo class to access simulation data
info = SimInfo()
