from ac_controller import ACController
from ac_socket import ACSocket

def main():
    """
    Main Function
    """
    controller = ACController()
    sock = ACSocket()

    with sock.connect() as conn:
        while(1):
            sock.update()
            try:
                # Convert the byte data to a string
                data_str = sock.data.decode('utf-8')

                # Split the string by commas and map values to a dictionary
                data_dict = dict(map(lambda x: x.split(':'), data_str.split(',')))
                print(data_dict)
            except:
                # If the data is invalid, throw error and return empty dict
                print(("Error parsing data, returning empty dict!", "red"))
                return {}
            try:
                controller.perform(0.5, 0)
                print("Controller input sent")
            except Exception as e:
                print(f"Error sending input: {e}")

if __name__ == "__main__":
    """
    Run the main function.
    """
    main()