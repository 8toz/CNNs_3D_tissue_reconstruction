import utils._image_operations as _image_operations

import os
import cv2
from patchify import patchify

def patchify_large_labels(directory, to_patchify=["adipocytes", "stroma"]):
    """
    Given a to_patchify list we check the image size and slice it into chunks
    to improve the mosaic fill afterwards
    """
    for img_path in os.listdir(directory):
        #print(directory)
        full_file_path = os.path.join(directory, img_path)
        
        patchified = False
        # Checks if the label is from the group we want to patchify
        if any([label in img_path for label in to_patchify]):
            img = cv2.imread(full_file_path)
            if img.shape[0] > 128 and img.shape[1] > 128:
                #print(full_file_path)
                patches = patchify(img, (64, 64, 3), step=64)
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        # Get a single patch
                        single_patch = patches[i, j, 0, :, :, :]
                        cv2.imwrite(os.path.join(directory, str(i)+str(j)+"_patchified_"+str(img_path)), single_patch)
                        patchified = True
                print("image",full_file_path, "patchified")

        if patchified:
            print("Removing", full_file_path)
            os.remove(full_file_path)

    check_label_fill_and_remove(directory, to_check_and_remove=to_patchify)

def check_label_fill_and_remove(directory, to_check_and_remove=["adipocytes", "stroma"]):
    """
    Check labels from directory if the dark background is dominant removes it from the folder
    """
    for img_path in os.listdir(directory):
        if any([label in img_path for label in to_check_and_remove]) and ("patchified" in img_path):
            full_file_path = os.path.join(directory, img_path)
            if not _image_operations.is_image_squared_with_non_black_pixels(image_path=full_file_path, tol_nonzero_pixels=0.25):
                #print("Removing", full_file_path)
                os.remove(full_file_path) # If the difference of label vs background is large removes the label
