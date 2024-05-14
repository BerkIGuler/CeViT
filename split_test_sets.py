import os
import shutil

# Set the directory where your files are located
source_dir = 'mismatched_test_dataset'

organize_by = "SNR-"


# Function to create a directory if it does not exist
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Get all files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Iterate over the files
for file in files:
    # Split the file name to extract the DS value
    parts = file.split('_')
    for part in parts:
        if part.startswith(organize_by):
            value = part.split('-')[1]
            break

    # Create the destination directory path
    dest_dir = os.path.join(source_dir, value)

    # Create the directory if it does not exist
    create_dir_if_not_exists(dest_dir)

    # Move the file to the destination directory
    shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir, file))

print(f"Files have been organized by {organize_by} value.")
