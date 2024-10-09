import glob
import os

def list_subdirectories(directory):
    # Use glob to get all entries in the directory
    all_entries = glob.glob(os.path.join(directory, "*"))

    # Filter out only the directories
    subdirectories = [entry for entry in all_entries if os.path.isdir(entry)]

    return subdirectories