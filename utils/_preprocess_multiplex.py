import utils._file_operations

import os
import glob
import re
import shutil
import pandas as pd
import numpy as np
import napari

from tqdm.notebook import tqdm_notebook
from natsort import natsorted

os.environ["OPENCV_IO_MAX_result_PIXELS"] = pow(2,40).__str__()

import cv2


def blend_multiplex(directory, invert_image=False):
    blended_multiplex_path = "./data/blended_multiplex"
    if os.path.exists(blended_multiplex_path):
        shutil.rmtree(blended_multiplex_path)
    utils._file_operations.create_folder(blended_multiplex_path)

    img_dir = os.path.join(directory, "*.tif") 
    paths = glob.glob(img_dir)
    
    # Regular expression to match Set*_***
    pattern = re.compile('[Ss]et\d+[-_]\d+')

    unique_sets = set()

    for path in paths:
        match = pattern.search(path)
        if match:
            unique_sets.add(match.group())

    unique_sets_list = sorted(unique_sets)

    for unique_set in tqdm_notebook(unique_sets_list):
        pattern = os.path.join(directory, "*" + unique_set + "*.tif")
        channels_path = glob.glob(pattern)

        assert len(channels_path) == 8, "Something went wrong with the Multiplex files chech the corresponding directory"

        
        # Load the images in grayscale
        ch1 = cv2.imread(channels_path[0], cv2.IMREAD_GRAYSCALE)
        ch2 = cv2.imread(channels_path[1], cv2.IMREAD_GRAYSCALE)
        ch3 = cv2.imread(channels_path[2], cv2.IMREAD_GRAYSCALE)
        ch4 = cv2.imread(channels_path[3], cv2.IMREAD_GRAYSCALE)
        ch5 = cv2.imread(channels_path[4], cv2.IMREAD_GRAYSCALE)
        ch6 = cv2.imread(channels_path[5], cv2.IMREAD_GRAYSCALE)
        ch7 = cv2.imread(channels_path[6], cv2.IMREAD_GRAYSCALE)
        ch8 = cv2.imread(channels_path[7], cv2.IMREAD_GRAYSCALE)
        # Define and apply Colormaps
        ch1_colormap = cv2.applyColorMap(ch1, cv2.COLORMAP_OCEAN)
        ch2_colormap = cv2.applyColorMap(ch2, cv2.COLORMAP_BONE)
        ch3_colormap = cv2.applyColorMap(ch3, cv2.COLORMAP_MAGMA)
        ch4_colormap = cv2.applyColorMap(ch4, cv2.COLORMAP_INFERNO)
        ch5_colormap = cv2.applyColorMap(ch5, cv2.COLORMAP_DEEPGREEN)
        ch6_colormap = cv2.applyColorMap(ch6, cv2.COLORMAP_PINK)
        ch7_colormap = cv2.applyColorMap(ch7, cv2.COLORMAP_OCEAN)
        ch8_colormap = cv2.applyColorMap(ch8, cv2.COLORMAP_OCEAN)
        # Blend Images
        step1 = cv2.add(ch1_colormap, ch2_colormap)
        step2 = cv2.add(step1, ch3_colormap)
        step3 = cv2.add(step2, ch4_colormap)
        step4 = cv2.add(step3, ch5_colormap)
        step5 = cv2.add(step4, ch6_colormap)
        step6 = cv2.add(step5, ch7_colormap)
        blended_image = cv2.add(step6, ch8_colormap)
        #color_blended_image = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)

        if invert_image:
            "Success" if cv2.imwrite(os.path.join(blended_multiplex_path, unique_set.split("_")[-1] +".tif"), (255-blended_image)) else print("Something went wrong saving the image")
        else:
            "Success" if cv2.imwrite(os.path.join(blended_multiplex_path, unique_set.split("_")[-1] +".tif"), blended_image) else print("Something went wrong saving the image")


def group_multiplex_files():

    files = natsorted(glob.glob("./data/blended_multiplex/*.tif"))

    section_counter = 1
    grouped_multiplex_path = "./data/grouped_multiplex"
    last_slide = int(os.path.basename(files[0]).split(".")[0])
    distance_between_multiplex = 7

    if os.path.exists(grouped_multiplex_path):
        shutil.rmtree(grouped_multiplex_path)
    utils._file_operations.create_folder(grouped_multiplex_path)

    section_path = os.path.join(grouped_multiplex_path, "section_"+str(section_counter))
    utils._file_operations.create_folder(section_path)

    for file in files:
        current_slide = int(os.path.basename(file).split(".")[0])
        if len(os.listdir(section_path)) == 4 or current_slide >= last_slide + distance_between_multiplex:
            print("Group detected from ", os.listdir(section_path)[0], "to", os.listdir(section_path)[-1])
            section_counter += 1
            section_path = os.path.join(grouped_multiplex_path, "section_"+str(section_counter))
            utils._file_operations.create_folder(section_path)
        
        
        shutil.copy(file, section_path)
        last_slide = int(os.path.basename(file).split(".")[0])

    print("Group detected from ", os.listdir(section_path)[0], "to", os.listdir(section_path)[-1])
    print("------------------------------------------")
    print("Inserting the HE reference for each group")
    # Look for registered HE and copy it inside each grouped section
    he_aligned_images = glob.glob("./data/66-4/processed_32/clean_dust_bubbles/registered/elastic registration/*.tif")

    for group in natsorted(os.listdir("./data/grouped_multiplex")):
        multiplex_subgroup = glob.glob(os.path.join("./data/grouped_multiplex/", group, "*.tif"))
        min_slice = int(os.path.basename(multiplex_subgroup[0]).split(".")[0])
        max_slice = int(os.path.basename(multiplex_subgroup[-1]).split(".")[0])
        # Trigger to avoid running the code again once the HE image is copied
        if len(multiplex_subgroup) > 4:
            break
        for file in he_aligned_images:
            match = re.search(r'_(\d+)_', file)
            number = match.group(1)
            if int(number) > min_slice and int(number) < max_slice:
                print(group, file)
                shutil.copy(file, os.path.join("./data/grouped_multiplex", group, str(number)+".tif"))
                break


import scipy.io

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def get_affine_dataframe():

    df = pd.DataFrame(columns=["szz", "padall", "cent", "f", "krf", "affine2d"])

    for i, group in enumerate(natsorted(os.listdir("./data/grouped_multiplex"))):
        files = glob.glob("./data/grouped_multiplex/"+ group + "/registered/elastic registration/save_warps/*.mat")

        for file in files: 
            filename = get_file_name(file)
            
            rotation_path = glob.glob("./data/warps/"+str(filename)+".mat")
            if len(rotation_path) != 0:
                rotation_file = rotation_path[0]
                rotations_dict = {}

                mat = scipy.io.loadmat(file)

                rotations_dict["szz"] = mat["szz"]
                rotations_dict["padall"] = mat["padall"]
                rotations_dict["cent"] = mat["cent"]
                rotations_dict["f"] = mat["f"]
                rotations_dict["krf"] = mat["krf"]

                mat = scipy.io.loadmat(rotation_file)
                rotations_dict["affine2d"] = mat["T_matrix"]

                # Convert dictionary to DataFrame row
                row = {key: [value] for key, value in rotations_dict.items()}

                # Append the row to the DataFrame
                df = pd.concat([df, pd.DataFrame(row, index=[filename])])
            else: 
                print(file, "HE does not have rotation")
    
    return df


def apply_affine_channel_wise(dir):

    df = get_affine_dataframe()

    for file in tqdm_notebook(glob.glob("./data/multiplex/processed_32/*.tif")):
        match = re.search(r'_(\d+)_Scan', file)
        number = match.group(1)

        img = cv2.imread(file)
        print(file)

        # Define the center point
        center_point = df.loc[number]["cent"][0]

        # Extract the affine transformation matrix
        matrix = df.loc[number]["affine2d"].T[:2]

        # Ensure the matrix is a NumPy array and has the correct shape
        matrix = np.array(matrix, dtype=np.float32)

        # Get dimensions of both images
        orig_height, orig_width = img.shape[:2]
        ref_height, ref_width = df.loc[number]["szz"][0]

        # Calculate padding needed
        pad_top = (ref_height - orig_height) // 2
        pad_bottom = ref_height - orig_height - pad_top
        pad_left = (ref_width - orig_width) // 2
        pad_right = ref_width - orig_width - pad_left

        # Pad the original image with black (0, 0, 0)
        padded_image = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Adjust the affine matrix to consider the center point
        # Translate the center point to the origin
        translation_to_origin = np.array([
            [1, 0, -center_point[0]],
            [0, 1, -center_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Translate back to the center point
        translation_back = np.array([
            [1, 0, center_point[0]],
            [0, 1, center_point[1]],
            [0, 0, 1]
        ], dtype=np.float32)

        # Combine the transformations
        affine_matrix_3x3 = np.vstack([matrix, [0, 0, 1]])  # Convert to 3x3 matrix for combination

        adjusted_matrix = np.dot(np.dot(translation_back, affine_matrix_3x3), translation_to_origin)

        # Extract the 2x3 part of the adjusted matrix for cv2.warpAffine
        adjusted_matrix_2x3 = adjusted_matrix[:2]

        # Apply the transformation
        transformed_image = cv2.warpAffine(padded_image, adjusted_matrix_2x3, (padded_image.shape[1], padded_image.shape[0]))

        file_name = os.path.basename(file)

        cv2.imwrite("./data/registered_multiplex/"+file_name, transformed_image)

def visualize_multiplex():
    # TODO check length if empty raise error
    
    multiplex_files = glob.glob("./data/blended_multiplex/*.tif")

    multiplex_numbers = []

    for file in multiplex_files: 
        multiplex_numbers.append(os.path.basename(file).split(".")[0]) 

    multiplex_stack = {"DAPI":[], "Opal_480":[], "Opal_520":[], "Opal_570":[], "Opal_620":[], "Opal_690":[], "Opal_780":[], "Sample_AF":[]}
    counter = 0

    for channel in tqdm_notebook(multiplex_stack):
        for file_num in multiplex_numbers:
            for file in glob.glob("./data/registered_multiplex/*.tif"):
                if channel in file and file_num in file:
                    counter += 1
                    image = cv2.imread(file, 0)
                    resized = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
                    multiplex_stack[channel].append(resized)
                    # if counter == 4:
                    #     for i in range(7):
                    #         multiplex_stack[channel].append(np.zeros_like(resized))
                    #     counter = 0



    colormaps = ["blue", "bop orange", "bop purple", "cyan", "gray", "green", "yellow", "gray"]

    viewer = napari.Viewer()
    for channel, color in zip(multiplex_stack, colormaps):
        stack = multiplex_stack[channel]
        stack = np.array(stack)
        viewer.add_image(stack, name=channel, scale=(20,8,8), colormap=color, blending="additive")
                        