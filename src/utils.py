import os

def create_data_directory(data_dir: str):
    """
    Create the data directory if it doesn't exist.
    """
    os.makedirs(data_dir, exist_ok=True)
