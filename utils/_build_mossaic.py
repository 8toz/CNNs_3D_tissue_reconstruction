import utils._file_operations as _file_operations
import utils._config as _config
import utils._image_operations as _image_operations
import utils._patchify_large_labels as _patchify_large_labels
import utils._data_augmentation as _data_augmentation

import numpy as np
import os
import random
from tqdm.notebook import tqdm_notebook

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2


def build_mossaic(labels_source_directory, mossaic_destination_directory, matrix_size=1024, data_augmentation=False, patchify_large_labels=False):
    """
    Parent method that runs:
        - Prepares the data in a dummy folder and process the labels with:
            cropping (patchify_large_labels) and data augmentation (augment_labels))
        - Get_extracted_labels_paths
        - Prepare_labels_masks
        - Create_mask_panel
    """
    _destination_path = _config.LABELS_DUMMY_DESTINATION_PATH

    if data_augmentation:
        print("Starting data augmentation")
        _data_augmentation.augment_labels(directory=labels_source_directory)

    _file_operations.prepare_mask_filling(src_folder=labels_source_directory, dest_folder=_destination_path) 

    if patchify_large_labels:
        print("Starting patchify large labels")
        _patchify_large_labels.patchify_large_labels(directory=_destination_path, to_patchify=["adipocytes", "stroma"])
    

    # Cleans folder with mossaics
    _file_operations.remove_files(mossaic_destination_directory)
    iter_number = 0 # Id of saved images
    
    while len(os.listdir(_destination_path)) != 0:
        files_before_processing = len(os.listdir(_destination_path))
        img_paths = get_extracted_labels_paths(labels_directory=_destination_path)
        rgb_list, mask_list, shuffled_image_paths = prepare_labels_masks(img_paths)
        create_mask_panel(mossaic_destination_directory, rgb_list, mask_list, shuffled_image_paths, iter_number, matrix_size)
        files_after_processing = len(os.listdir(_destination_path))
        if files_before_processing == files_after_processing:
            print("The remaining images do not fit into the matrix size, try increasing it or reduce label size")
            break
        iter_number += 1

    print(f"{str(iter_number)} images processed successfully")


# Maybe use glob to simplify this method
def get_extracted_labels_paths(labels_directory):
    """
    Given a folder directory gets all its png's image paths 
    """
    image_paths = []

    for root, dirs, files in os.walk(labels_directory):
        for file in files:
            if file.lower().endswith(('.png')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)

    return image_paths

def prepare_labels_masks(image_files, seed=32):
    """
    Given the paths from (get_extracted_labels_paths) creates 
    a list with all the images and another one with its 
    corresponding masks.
    """
    mask_list = []
    rgb_list = []

    # We shuffle the image files
    random.Random(seed).shuffle(image_files)

    shuffled_image_paths = image_files
    print(f"Labels remaining: {len(shuffled_image_paths)}")

    for f in shuffled_image_paths:
        image = cv2.imread(f)
        rgb_list.append(image)

        # Convert the color image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold the grayscale image to get a binary mask of non-black pixels
        _, mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        # Convert the binary mask to a matrix with labels
        labels_matrix = np.where(mask > 0, _config.get_label_index(f), 0)
        mask_list.append(labels_matrix)

    return rgb_list, mask_list, shuffled_image_paths

def create_mask_panel(mossaic_destination_directory, rgb_list, mask_list, shuffled_image_paths, iter_number, matrix_size=1024):
    """
    Given the masks and images fills a dark image with the labels
    without overlap
    """
    
    # Initialize the global image matrix
    global_mask = np.zeros((matrix_size, matrix_size))
    global_rgb = np.zeros((matrix_size, matrix_size, 3), dtype=np.uint8)

    check = 0
    x_mark, y_mark = 0, 0

    # Iterate over each mask
    for pointer, labels_matrix in enumerate(mask_list):
        # x_mark, y_mark = 0, 0 # This takes too long to converge
        if check == 1:
            break
        
        # Iterate over rows of the global image
        while x_mark + labels_matrix.shape[0] < global_mask.shape[0]:
            # Check if the mask fits horizontally without going out of bounds
            if y_mark + labels_matrix.shape[1] > global_mask.shape[1]:
                # Move to the next row
                x_mark += 1
                y_mark = 0
                
                continue

            # Check if the current position in the global image is empty
            if np.all(global_mask[x_mark:x_mark+labels_matrix.shape[0], y_mark:y_mark+labels_matrix.shape[1]] == 0):
                
                # Place the mask into the global image
                global_mask[x_mark:x_mark+labels_matrix.shape[0], y_mark:y_mark+labels_matrix.shape[1]] += labels_matrix
                
                global_rgb[x_mark:x_mark+labels_matrix.shape[0], y_mark:y_mark+labels_matrix.shape[1]] += rgb_list[pointer]
                os.remove(shuffled_image_paths[pointer])
                break

            if x_mark >= matrix_size and y_mark >= matrix_size:
                check = 1
                break
            # Move to the next column
            y_mark += 1

    white_bg_global_rgb = _image_operations.change_black_to_white(global_rgb)
        
    cv2.imwrite(os.path.join(mossaic_destination_directory, "rgb_sample_" + str(iter_number) + ".png"), white_bg_global_rgb)
    print("rgb_sample_" + str(iter_number) + ".png")
    cv2.imwrite(os.path.join(mossaic_destination_directory, "mask_sample_" + str(iter_number) + ".png"), global_mask)
    print("mask_sample_" + str(iter_number) + ".png")




