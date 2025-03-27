import time  # Import the time module to manage time-related tasks
import vgamepad  # Import the vgamepad library to simulate a virtual game controller
import keyboard  # Import the keyboard library to simulate keyboard inputs
import numpy as np  # Import NumPy for numerical operations (though not used in the provided code)


class ACController:
    """
    A virtual controller for Assetto Corsa.
    This class uses the vgamepad library to send inputs to the game.
    """

    def __init__(self):
        """
        Initialize the virtual controller.
        This method sets up the virtual gamepad and simulates a button press
        to ensure that Assetto Corsa recognizes the controller.
        """
        # Create an instance of the virtual gamepad
        self.gamepad = vgamepad.VX360Gamepad()

        # Press and release a button (A button) so AC recognizes the controller
        self.gamepad.press_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)  # Press the A button
        self.gamepad.update()  # Update the gamepad state to reflect the button press
        time.sleep(0.5)  # Wait for half a second
        self.gamepad.release_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)  # Release the A button
        self.gamepad.update()  # Update the gamepad state to reflect the button release

    def perform(self, throttle_brake, steer):
        """
        Perform the actions in the game based on input values.
        :param throttle_brake: The throttle value (positive for throttle, negative for brake).
        :param steer: The steering value (left/right).
        """
        # Calculate throttle and brake values, ensuring they are non-negative
        throttle = max(0.0, throttle_brake)  # Throttle is positive
        brake = max(0.0, -throttle_brake)  # Brake is positive when throttle_brake is negative
        
        # Set the left trigger (brake) and right trigger (throttle) values
        self.gamepad.left_trigger_float(value_float=brake)  # Set brake value
        self.gamepad.right_trigger_float(value_float=throttle)  # Set throttle value
        
        # Set the left joystick position for steering, y-value is set to 0 for no vertical movement
        self.gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
        
        # Update the gamepad state to send the new input values to the game
        self.gamepad.update()

    def reset_car(self):
        """
        Reset the car back to the starting line in the game.
        This method simulates pressing the F10 key to respawn the car
        and then moves the car slightly to ensure it is positioned correctly.
        """
        # Simulate pressing the F10 key on the keyboard to trigger a respawn in the AC app
        keyboard.press('F10')  # Press the F10 key
        time.sleep(0.5)  # Wait for half a second to ensure the key press is registered
        keyboard.release('F10')  # Release the F10 key

        # Move the car forward a little bit so it's over the starting line
        self.perform(0.0, 0.0)  # Perform no throttle or steering to stabilize the car
        time.sleep(0.4)  # Wait for a short duration
        self.perform(1.0, 0.0)  # Apply full throttle to move the car forward
        time.sleep(2.4)  # Wait for the car to move forward for a while
        self.perform(-1.0, 0.0)  # Apply full reverse to move the car back slightly
        time.sleep(1.4)  # Wait for the car to move back
        self.perform(0.0, 0.0)  # Stop the car
        time.sleep(0.3)  # Wait for a short duration before ending the reset
