from ac_controller import ACController

def main():
    """
    Main Function
    """
    controller = ACController()

    while(1):
        controller.perform(0.5, 0.5)

if __name__ == "__main__":
    """
    Run the main function.
    """
    main()