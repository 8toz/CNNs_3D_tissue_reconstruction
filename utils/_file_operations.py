import os 
import shutil
import glob

def prepare_mask_filling(src_folder, dest_folder, remove=True):
    '''
    Moves all the data from one folder to another. If files are already copied you can use remove=False 
    to avoid this step and only get the files out of folders
    '''
    if remove:
        remove_files(dest_folder) # Clean destination folder
        copy_folder(src_folder, dest_folder) # Copies folder data to prevent losing all original labels
    # Moves all files ouside the subfolder dir to facilitate mask fill
    for folder in os.listdir(dest_folder):
        src_folder = os.path.join(dest_folder, folder)
        # Check if it's a directory
        if os.path.isdir(src_folder):
            # Move files from source folder to destination folder
            for filename in os.listdir(src_folder):
                src_file = os.path.join(src_folder, filename)
                dest_file = os.path.join(dest_folder, filename)
                shutil.move(src_file, dest_file)
            # Remove the source folder
            shutil.rmtree(src_folder)

    print("All files moved and folders removed successfully.")

def clean_images_from_directory(directory_path):
    """
    Checks for all png images in a directory and removes them
    """
    files = os.listdir(directory_path)
    png_files = [file for file in files if file.endswith('.png')]

    for png_file in png_files:
        os.remove(os.path.join(directory_path, png_file))

    print("Images Cleaned from", directory_path)


def get_tif_paths(path, resolution):
    '''
    Searchs given a path for all the processed TIF files in a folder
    '''
    path_list = glob.glob(os.path.join(path, "processed_"+str(resolution), "*.tif"))
    return path_list

def get_labelled_files(labels_path, file_list):
    '''
    Returns a tuple with the image path and the label associated with it
    '''
    file_label_list = []
    if len(file_list) == 0:
        raise("Create preprocessed files for this resolution")
    
    labelled_files = os.listdir(labels_path)
    for file in file_list:
        file_name = os.path.splitext(os.path.basename(file))[0]
        for labelled_file in labelled_files:
            if file_name in labelled_file:
                file_label_list.append((file, os.path.join(labels_path, labelled_file)))

    return file_label_list



def create_folder(folder_path):
    '''
    Creates a folder in the requested path if it does not already exists
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass

def remove_files(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Iterate over the files and remove each one
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print(f"Files removed from {directory}")

def copy_folder(src_folder, dest_folder):
    try:
        # Copy the entire folder structure and its files recursively
        shutil.copytree(src_folder, dest_folder, dirs_exist_ok = True)
        print(f"Folder '{src_folder}' copied to '{dest_folder}' successfully.")
    except Exception as e:
        print(f"Error copying folder: {e}")

def move_files(src_folder, dest_folder):
    try:
        # Create the destination folder if it doesn't exist
        os.makedirs(dest_folder, exist_ok=True)
        
        # Iterate over all files in the source folder
        for filename in os.listdir(src_folder):
            # Get the full path of the source file
            src_file = os.path.join(src_folder, filename)
            # Get the full path of the destination file
            dest_file = os.path.join(dest_folder, filename)
            # Move the file to the destination folder
            shutil.move(src_file, dest_file)
        print(f"All files moved from '{src_folder}' to '{dest_folder}' successfully.")
    except Exception as e:
        print(f"Error moving files: {e}")