import os
import shutil

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def move_files_to_directory(source_dir, dest_dir):
    create_directory_if_not_exists(dest_dir)
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        dest_file = os.path.join(dest_dir, filename)
        shutil.move(source_file, dest_file)
