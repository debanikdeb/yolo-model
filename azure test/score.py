# Import both init and run from inference.py
from inference import init, run as inference_run

# Call the init() function to initialize the model
init()

# Define the entry point function for requests


def run(input_data):
    # Call the 'run' function from inference.py
    return inference_run(input_data)
