import os

"""
    Returns a file in the directory "data".

    :param file_name: File name in the directory "data".
"""
def get_file(file_name):
    current_directory = os.getcwd()
    return os.path.join(current_directory, 'data', file_name)