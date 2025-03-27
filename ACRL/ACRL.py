import sys
import os
import platform
import socket
import threading
import ac_api.car_info as ci
import ac_api.input_info as ii
import ac_api.lap_info as li

# The name of the app (ACRL: Assetto Corsa Reinforcement Learning)
APP_NAME = 'ACRL'

# Add the third party libraries to the path
try:
    # Determine if the architecture is 64-bit or not and set the appropriate directory
    if platform.architecture()[0] == "64bit":
        sysdir = "stdlib64"
    else:
        sysdir = "stdlib"
    
    # Insert the third-party libraries path into system path
    sys.path.insert(len(sys.path), 'apps/python/{}/third_party'.format(APP_NAME))
    os.environ['PATH'] += ";."
    sys.path.insert(len(sys.path), os.path.join('apps/python/{}/third_party'.format(APP_NAME), sysdir))
    os.environ['PATH'] += ";."
except Exception as e:
    # Log any errors encountered during the import of libraries
    ac.log("[ACRL] Error importing libraries: %s" % e)

import ac  # Import the Assetto Corsa API
from IS_ACUtil import *  # Import utility functions (assumed to be defined in IS_ACUtil)

# Training enabled flag
training = False
completed = False

# Socket variables for communication
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65431  # The port used by the server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP socket
connected = False  # Connection status
# Socket thread for handling communication
t_sock = None

# Respawn thread for handling car respawn functionality
t_res = None
RES_KEY = 121  # Key code for F10

# UI label and button variables
label_model_info = None
btn_start = None

def acMain(ac_version):
    """
    The main function of the app, called on app start.
    :param ac_version: The version of Assetto Corsa as a string.
    """
    global label_model_info, btn_start, t_res  # Declare global variables
    ac.console("[ACRL] Initializing...")  # Log initialization message

    # Create the app window
    APP_WINDOW = ac.newApp(APP_NAME)  # Create a new app window
    ac.setSize(APP_WINDOW, 320, 140)  # Set the size of the app window
    ac.setTitle(APP_WINDOW, APP_NAME + ": Reinforcement Learning")  # Set the title of the app

    # Set the background of the app window to fully black
    ac.setBackgroundOpacity(APP_WINDOW, 1)

    # Create an info label to display training status
    label_model_info = ac.addLabel(APP_WINDOW, "Training: " + str(training) + "\nClick start to begin!")
    ac.setPosition(label_model_info, 320/2, 40)  # Center the label
    ac.setFontAlignment(label_model_info, "center")  # Center the text

    # Create a start button for initiating training
    btn_start = ac.addButton(APP_WINDOW, "Start Training")
    ac.setPosition(btn_start, 20, 90)  # Position the button
    ac.setSize(btn_start, 280, 30)  # Set the size of the button
    ac.addOnClickedListener(btn_start, start)  # Add a listener for button clicks
    ac.setVisible(btn_start, 1)  # Make the button visible

    # Start the respawn listener thread
    t_res = threading.Thread(target=respawn_listener)  # Create a thread for respawn listening
    t_res.start()  # Start the thread

    ac.console("[ACRL] Initialized")  # Log completed initialization
    return APP_NAME  # Return the app name

def acUpdate(deltaT):
    """
    The update function of the app, called every frame.
    :param deltaT: The time since the last frame as a float.
    """
    # Update the model info label based on training status
    global completed, training
    if completed:
        ac.setText(label_model_info, "Training completed!" + "\nRestart to train again!")  # Update label if training is completed
    else:
        ac.setText(label_model_info, "Training: " + str(training))  # Update label with current training status

    if ac.getCameraMode() is not 4:
        # Lock the camera mode to helicopter view (mode 4)
        ac.setCameraMode(4)

def acShutdown():
    """
    The shutdown function of the app, called on app close.
    """
    global training
    training = False  # Disable training
    try:
        stop()  # Stop the training process
        t_res.join()  # Wait for the respawn thread to finish
        t_sock.join()  # Wait for the socket thread to finish
    except:
        pass  # Ignore any exceptions during shutdown
    ac.console("[ACRL] Shutting down...")  # Log shutdown message

def start(*args):
    """
    The function called when the start button is pressed.
    :param args: The arguments passed to the function.
    """
    global btn_start, training, connected, t_sock
    if not connect():  # Attempt to connect to the socket
        ac.console("[ACRL] Didn't start model, could not connect to socket!")  # Log connection failure
        connected = False  # Update connection status
        training = False  # Ensure training is disabled
        return

    ac.console("[ACRL] Starting model...")  # Log model start

    ac.setVisible(btn_start, 0)  # Hide the start button
    training = True  # Enable training

    # Start the socket listener thread if it hasn't been started yet
    if t_sock is None:
        t_sock = threading.Thread(target=sock_listener)  # Create a thread for socket listening
    t_sock.start()  # Start the thread

def stop(*args):
    """
    The function called when the training has stopped.
    :param args: The arguments passed to the function.
    """
    global btn_start, training, sock, connected, t_sock, completed

    ac.console("[ACRL] Stopping model...")  # Log model stop
    sock.close()  # Close the socket connection
    connected = False  # Update connection status
    training = False  # Disable training
    completed = True  # Mark training as completed

def connect():
    """
    Attempts to connect to the socket server.
    """
    global sock, connected
    if connected:  # If already connected, return true
        return True
    try:
        sock.connect((HOST, PORT))  # Attempt to connect to the server
        connected = True  # Update connection status
        ac.console("[ACRL] Socket connection successful!")  # Log successful connection
        return True
    except:
        ac.console("[ACRL] Socket could not connect to host...")  # Log connection failure
        return False

def respawn_listener():
    """
    Listens for a particular key press and will respawn the car at the finish line when pressed.
    """
    global completed
    while not completed:  # Continue listening until training is completed
        if getKeyState(RES_KEY):  # Check if the respawn key is pressed
            ac.console("[ACRL] Respawning...")  # Log respawn action
            sendCMD(68)  # Send command to restart to session menu
            sendCMD(69)  # Send command to start the lap + driving

def sock_listener():
    """
    Listens for signals from the socket during training.
    """
    global sock, training
    while True:
        # If not training, exit the loop
        if not training:
            break

        # If the socket is not connected, try to connect
        if not connect():
            ac.console("[ACRL] Socket could not connect to host in acUpdate, stopping training!")  # Log connection failure
            stop()  # Stop training
            break

        # Receive signals from the socket
        data = sock.recv(1024)  # Receive data from the socket
        if not data:  # If no data is received
            ac.console("[ACRL] Received stop signal, stopping training...")  # Log stop signal
            stop()  # Stop training
            break
        ac.console("[ACRL] Received request, responding with game data...")  # Log data request

        # Get the data from the game
        track_progress = ci.get_location()  # Get the car's track progress
        speed_kmh = ci.get_speed()  # Get the car's speed in km/h
        world_loc = ci.get_world_location()  # Get the car's world location
        throttle = ii.get_gas_input()  # Get the throttle input
        brake = ii.get_brake_input()  # Get the brake input
        steer = ii.get_steer_input()  # Get the steering input
        lap_time = li.get_current_lap_time()  # Get the current lap time
        lap_invalid = li.get_invalid()  # Check if the lap is invalid
        lap_count = li.get_lap_count()  # Get the lap count
        velocity = ci.get_velocity()  # Get the car's velocity

        # Turn the data into a string for transmission
        data = "track_progress:" + str(track_progress) + "," + \
               "speed_kmh:" + str(speed_kmh) + "," + \
               "world_loc[0]:" + str(world_loc[0]) + "," + \
               "world_loc[1]:" + str(world_loc[1]) + "," + \
               "world_loc[2]:" + str(world_loc[2]) + "," + \
               "throttle:" + str(throttle) + "," + \
               "brake:" + str(brake) + "," + \
               "steer:" + str(steer) + "," + \
               "lap_time:" + str(lap_time) + "," + \
               "lap_invalid:" + str(lap_invalid) + "," + \
               "lap_count:" + str(lap_count) + "," + \
               "velocity[0]:" + str(velocity[0]) + "," + \
               "velocity[1]:" + str(velocity[1]) + "," + \
               "velocity[2]:" + str(velocity[2])

        # Send the data in bytes format
        sock.sendall(str.encode(data))  # Send the constructed data to the socket
