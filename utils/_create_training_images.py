import utils._file_operations as _file_operations
import utils._image_operations as _image_operations

import re
import os
import cv2

import numpy as np

def prepare_training_images(mossaic_directory, destination_directory, slice_size=(256, 256), desired_background_color = [240, 239, 241], tolerance=0.05):
    images_dir = os.path.join(destination_directory, "images")
    masks_dir = os.path.join(destination_directory, "masks")

    rgb_ext = mossaic_directory + "/rgb_sample_"
    mask_ext = mossaic_directory + "/mask_sample_"

    _file_operations.clean_images_from_directory(images_dir)
    _file_operations.clean_images_from_directory(masks_dir)

    text = " ".join(os.listdir(mossaic_directory))

    # Find all digits
    pattern = r'(\d+)'
    matches = re.findall(pattern, text)
    max_image = max(map(int, matches))

    for img_number in range(max_image + 1):
        color_img = cv2.imread(rgb_ext + str(img_number) + ".png")
        mask_img = cv2.imread(mask_ext + str(img_number) + ".png", cv2.IMREAD_GRAYSCALE)
        slice_image(image=color_img, mask=mask_img, img_number=img_number, images_dir=images_dir, masks_dir=masks_dir, tolerance=tolerance, slice_size=slice_size)

    
    print(f"Setting background color to: {desired_background_color}")

    for img_path in os.listdir(images_dir):
        _image_operations.white_to_gray_BG(os.path.join(images_dir, img_path), desired_color=desired_background_color)

    print(f"Images ready for training at {images_dir}")

def slice_image(image, mask, img_number, images_dir, masks_dir, tolerance, slice_size):
    """
    Slice the rgb mossaic and the mask mossaic 
    for the training phase note that they should have
    the same names to keep image/mask pixel correspondence
    """



    height, width = image.shape[:2]

    for y in range(0, height, slice_size[1]):
        for x in range(0, width, slice_size[0]):
            # Extract the rgb
            rgb_img = image[y:y+slice_size[1], x:x+slice_size[0]]
            # Extract the mask
            mask_img = mask[y:y+slice_size[1], x:x+slice_size[0]]

            # Save the rgb slice
            name = f"img{img_number}_{y}_{x}.png"  # Same name for mask and image
            slice_path = os.path.join(images_dir, name)
            cv2.imwrite(slice_path, rgb_img)

            # Save the mask slice
            slice_path = os.path.join(masks_dir, name)
            cv2.imwrite(slice_path, mask_img)

    remove_non_square_images(images_dir, tolerance=tolerance, bool_mask=False)
    remove_non_square_images(masks_dir, tolerance=tolerance, bool_mask=True)

    if len(os.listdir(images_dir)) != len(os.listdir(masks_dir)):
        raise AssertionError(f"Number of images({len(os.listdir(images_dir))}) and masks ({len(os.listdir(masks_dir))}) do not match")

    return "Slicing completed!"

def remove_non_square_images(folder_path, tolerance, bool_mask):
    """
    Given a path checks all png images that are not 
    squared or contain all black pixel values removing
    them
    """
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png')):
            image_path = os.path.join(folder_path, file)
             # Check if the image is squared and has some nonzero pixels
            if bool_mask:
                if not _image_operations.is_image_squared_with_non_black_pixels(image_path, tol_nonzero_pixels=tolerance):
                    #print(image_path)
                    os.remove(image_path)
            else:
                if not _image_operations.is_image_squared_with_non_white_pixels(image_path, tol_white_pixels=tolerance):
                    #print(image_path)
                    os.remove(image_path)



