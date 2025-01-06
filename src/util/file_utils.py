import os

def get_file(file_name):
    current_directory = os.getcwd()
    return os.path.join(current_directory, 'data', file_name)