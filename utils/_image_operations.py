import os
import numpy as np
import glob

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

from tqdm import tqdm

from utils._file_operations import create_folder

def get_avg_height_width(directory):
    """
    Given a directory gets the average height and width of the images
    """
    height_list = []
    width_list = []

    files = glob.glob(directory)
    for file in files:
        img = cv2.imread(file)
        height_list.append(img.shape[0])
        width_list.append(img.shape[1])
    print(f"Average Height {sum(height_list)/len(height_list)} pixels.")
    print(f"Average Width {sum(height_list)/len(height_list)} pixels.")

def is_image_squared_with_non_white_pixels(image_path, tol_white_pixels=0.05):
    """
    Checks if the image is squared and there are labels present with a representative values (not all black)
    """
    if not os.path.isfile(image_path):
        print(f"File {image_path} does not exist.")
        return False

    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if the image is squared
    height, width = gray_img.shape
    if height != width:
        #print(f"Image {image_path} is not squared (height: {height}, width: {width}).")
        return False
    
    # Check if the image has at least 95% pixels by default white
    total_pixels = gray_img.size
    white_pixels = np.sum(gray_img == 255)
    white_ratio = white_pixels / total_pixels
    # Check if the white ratio is within the specified tolerance
    if white_ratio >= (1 - tol_white_pixels):
        #print(f"{white_ratio} white")
        return False
    
    return True

def is_image_squared_with_non_black_pixels(image_path, tol_nonzero_pixels=0.05):
    """
    Checks if the image is squared and there are labels present with a representative values (not all black)
    """
    if not os.path.isfile(image_path):
        #print(f"File {image_path} does not exist.")
        return False

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is square
    height, width = image.shape
    if height != width:
        # print(f"Image {image_path} is not square (height: {height}, width: {width}).")
        return False
    
    # Check if the image has at least min_nonzero_pixels different than 0
    nonzero_pixels = cv2.countNonZero(image)
    total_pixels = image.size
    black_ratio = (total_pixels-nonzero_pixels) / total_pixels

    if black_ratio >= (1-tol_nonzero_pixels):
        # print(f"{black_ratio} black")
        return False
    
    return True

def load_image(img_path):
    '''
    Loads an image from path
    '''
    image = cv2.imread(img_path)
    return image

## Maybe can be simplified
def change_black_to_white(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
    inverted_img = cv2.bitwise_not(binary_img)
    inverted_bgr_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    result_img = cv2.bitwise_or(img, inverted_bgr_img)

    return result_img

## TBP
def check_fill(masked_img, matrix_size):
    '''
    Checks how many pixels of the collage are labelled
    '''
    return sum(sum(np.where(masked_img>0, 1, 0)))/(matrix_size**2)

def white_to_gray_BG(img_path, desired_color=[240, 239, 241]):
    """ 
    Given a image path finds the white 
    pixels and replace them with the desired rgb color in place
    """
    img = cv2.imread(img_path)
    # Define the replacement color
    replacement_color = np.array(desired_color)
    # Find pixels with the value (255, 255, 255)
    white_pixels = np.all(img == [255, 255, 255], axis=-1)
   
    img[white_pixels] = replacement_color
    # Save the modified image
    cv2.imwrite(img_path, img)

def white_pad_image(image, divisor=128):
    """
    Pads an image with the required size to predict.
    """
    # Calculate the amount of padding needed
    height, width = image.shape[:2]
    pad_height = (divisor - height % divisor) % divisor
    pad_width = (divisor - width % divisor) % divisor

    # Create a white canvas with the desired dimensions
    white_canvas = np.ones((height + pad_height, width + pad_width, 3), dtype=np.uint8) * 255

    # Place the original image on the white canvas
    white_canvas[:height, :width] = image

    return white_canvas

def remove_dustNbubbles(directory, image_resolution=4):
    """
    Given a directory removes the bubbles from all the images within it
    and rewrites them in a new directory to avoid losing the original image.
    """
    
    robust_dilation_factor = int(32/image_resolution) # The pipeline was developed in x32 to correct for larger images we correct dilations
    create_folder(os.path.join(directory, "clean_dust_bubbles"))

    files =  glob.glob(os.path.join(directory, "*.tif"))

    for file in tqdm(files):
        file_name = file.split("\\")[-1] # Really specific for thesis  should be changed

        img = cv2.imread(file)
        clean_bubbles = bubble_removal(img, robust_dilation_factor)
        result = dust_removal(clean_bubbles, robust_dilation_factor)

        # Change data type to allow saving as tif file
        if cv2.imwrite(os.path.join(directory, "clean_dust_bubbles", file_name), result.astype("uint8")):
            #print(f"Success writing {file_name}")
            continue
        else:
            raise Exception("Failed to save images")
        

def bubble_removal(img, robust_dilation_factor, color=[240, 239, 241]):
    """
    Detects image bubbles and fills the space with the desired RGB color
    """
    img = img.astype("uint8")
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowRange = np.array([0, 0, 0])
    uppRange = np.array([179, 100, 100])
    mask = cv2.inRange(hsvImage, lowRange, uppRange)
    kernel = np.ones((3, 3), dtype='uint8')
    img_dilation = cv2.dilate(mask, kernel, iterations=20*robust_dilation_factor)
    contours, _ = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = cv2.drawContours(img_dilation, contours, -1, color=(255, 0, 0), thickness=cv2.FILLED)

    kernel = np.ones((3, 3), dtype='uint8')
    bubble_mask = cv2.erode(filled, kernel, iterations=15*robust_dilation_factor)

    bubble_mask = 255 - bubble_mask
    bubble_mask = cv2.merge([bubble_mask] * 3)
    bubbles_removed = np.where(bubble_mask == 255, img, color)  # [240, 239, 241]

    return bubbles_removed

def dust_removal(img, robust_dilation_factor, color=[240, 239, 241]):
    """
    Detects dust and fills the space with the desired RGB color
    """
    img = img.astype("uint8")
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowRange = np.array([0, 0, 0])
    uppRange = np.array([179, 275, 100])
    mask = cv2.inRange(hsvImage, lowRange, uppRange)
    kernel = np.ones((3, 3), dtype='uint8')
    img_erosion = cv2.erode(mask, kernel, iterations=1*robust_dilation_factor) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=10*robust_dilation_factor)
    dust_mask = 255 - img_dilation
    dust_mask = cv2.merge([dust_mask] * 3)
    dust_removed = np.where(dust_mask == 255, img, color)  

    return dust_removed