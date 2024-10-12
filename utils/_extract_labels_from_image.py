import utils._file_operations as _file_operations
import utils._config as _config
import utils._image_operations as _image_operations

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon
import glob
import shutil
import random

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

from tqdm.notebook import tqdm_notebook
from shapely.affinity import scale, translate

def generate_training_labels(images_path, labels_path, resolution, validation_split, test_split):
    '''
    Given the image path, the labels path and the desired resolution 
    (note the resolution for that image should already exist processed_XX folder).
    Extract the separated labels into the processed_labels_XX folder
    '''
    
    training_storage = "./data/processed_labels_"+str(resolution)
    _file_operations.create_folder(training_storage)

    # Clean directories iteratively
    for subdir in os.listdir(training_storage):
        _file_operations.remove_files(os.path.join(training_storage, subdir))

    _file_operations.create_folder(training_storage)
    [_file_operations.create_folder(os.path.join(training_storage,str(_config.LABEL_DICT[key])+"_"+key)) for key in _config.LABEL_DICT]

    files_list = _file_operations.get_tif_paths(images_path, resolution)
    file_label_list = _file_operations.get_labelled_files(labels_path, files_list)
    for img_path, label_path in file_label_list:
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        print("Extracting image ", image_name, "labels...")
        image = _image_operations.load_image(img_path)
        labels_df  = load_labels(label_path, int(resolution))
        
        
        # Apply function to extract label for each row
        for idx, row in tqdm_notebook(labels_df.iterrows()):
            extract_label(image, image_name, training_storage, row["encoded_label"], row["label"], row["geometry"], idx)
    
    train_test_split(labels_path=training_storage, validation_split=validation_split, test_split=test_split, seed=32)


def extract_label(image, image_name, training_storage, encoded_label, label, geometry, idx):#labels_df, idx):
    '''
    Returns a tuple with the image path and the label associated with it
    '''
    folder_name = str(encoded_label) + "_" + (label)
    polygon_gdf = geometry
    # Get the bounding box of the polygon
    min_x, min_y, max_x, max_y = np.floor(polygon_gdf.bounds).astype("int64")
    crop = image[min_y:max_y, min_x:max_x]

    translate_x = ((max_x-min_x) - polygon_gdf.bounds[0] - polygon_gdf.bounds[2]) / 2
    translate_y = ((max_y-min_y) - polygon_gdf.bounds[1] - polygon_gdf.bounds[3]) / 2
    translated_polygon = translate(polygon_gdf, xoff=translate_x, yoff=translate_y)

    contour = np.array([[int(x), int(y)] for x, y in translated_polygon.exterior.coords])

    mask = np.zeros_like(crop)*255
    cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
    masked_img = cv2.bitwise_and(crop, mask)

    cv2.imwrite(os.path.join(training_storage, folder_name, image_name + "_" + label+"_"+str(idx)+".png"), masked_img)
    
    return True

def load_labels(path, downscaling_factor=None):
    '''
    Given a labelled GeoJson loads it and returns it as a preprocessed Pandas dataframe
    '''
    to_drop = ["id", "objectType", "classification", "object_type", "isLocked"]
    labels_df = gpd.read_file(path)
    if len(labels_df[labels_df["classification"].isna()]) > 0:
        print("Droping contours without label", labels_df[labels_df["classification"].isna()].index.values)
        labels_df = labels_df.dropna(subset=["classification"])
    names = list(labels_df["classification"][0].keys())
    labels_df["label"] = labels_df["classification"].apply(lambda x: x[names[0]])
    #labels_df["color"] = labels_df["classification"].apply(lambda x: rgb_to_hex(x[encoded_names[1]]))
    labels_df = labels_df.drop(columns=to_drop, errors="ignore")
    labels_df["geometry"] = labels_df["geometry"].apply(lambda row: check_multipolygon(row))
    # Scale the coordinates
    if downscaling_factor is not None:
        labels_df['geometry'] = labels_df['geometry'].apply(lambda row: scale_geometry(row, (1/downscaling_factor)))

    # Encode and clean labels based in dict
    labels_df["label"] = labels_df["label"].str.lower()
    labels_df["label"] = labels_df["label"].str.replace(" ", "_")
    # Fix because labels are not consistent in the data
    labels_df['label'] = labels_df['label'].apply(lambda x: 'blood_vessels' if 'blood_vessel' in x else x)
    labels_df['label'] = labels_df['label'].apply(lambda x: 'adipocytes' if 'fat' in x else x)

    labels_df["encoded_label"] = labels_df["label"].apply(lambda x: _config.LABEL_DICT[x])

    return labels_df

def scale_geometry(geom, scaling_factor):
    '''
    Rescales a geometry object based on the image downscaling.
    Note that origin (0,0) allows to fix it to its correct position on the downscaled image.
    '''
    return scale(geom, xfact=scaling_factor, yfact=scaling_factor, zfact=1, origin=(0, 0))



def check_multipolygon(row):
    """
    Checks if the object is a Multipolygon if so returns its polygon with largest area
    """
    if isinstance(row, MultiPolygon):
        area = -1
        for poly in row.geoms:
            if poly.area > area:
                area = poly.area
                fixed_polygon = poly
                row = fixed_polygon
                return row
    else:
        return row
    
def train_test_split(labels_path, validation_split, test_split, seed=32):
    train_labels_dir = "./data/train_labels"
    val_labels_dir = "./data/validation_labels"
    test_labels_dir = "./data/test_labels"

    _file_operations.create_folder(train_labels_dir)
    _file_operations.create_folder(val_labels_dir)
    _file_operations.create_folder(test_labels_dir)

    # Clean directories iteratively
    for subdir in os.listdir(train_labels_dir):
        _file_operations.remove_files(os.path.join(train_labels_dir, subdir))

    for subdir in os.listdir(val_labels_dir):
        _file_operations.remove_files(os.path.join(val_labels_dir, subdir))
        
    for subdir in os.listdir(test_labels_dir):
        _file_operations.remove_files(os.path.join(test_labels_dir, subdir))

    for dir in os.listdir(labels_path):
        # Only iterates through the labels directories
        if dir[0].isdigit():
            imgs_paths = glob.glob(os.path.join(labels_path, dir, "*.png"))
            # Selects labels randomly from folders and add them to their corresponding subfolder
            # Shuffle operation is in place
            # Deterministic shuffling for reproducible results
            random.Random(seed).shuffle(imgs_paths)
            n_images = len(imgs_paths)
            val_images = int(np.ceil(validation_split * n_images))
            test_images = int(np.ceil(test_split * n_images))
            train_images = n_images - val_images - test_images
            assert(train_images+val_images+test_images == n_images)

            print(f"{dir}: \n", f"Train Split - {train_images} labels", f"\n Validation Split - {val_images}", f"\n Test Split - {test_images}")

            training_paths = imgs_paths[:train_images]
            validation_paths = imgs_paths[train_images:train_images + val_images]
            test_paths = imgs_paths[train_images+val_images:]

            # Copy desired files to training subfolder
            _file_operations.create_folder(os.path.join(train_labels_dir, dir))
            for file in training_paths:
                shutil.copy(file, os.path.join(train_labels_dir, dir))

            # Copy desired files to validation subfolder
            _file_operations.create_folder(os.path.join(val_labels_dir, dir))
            for file in validation_paths:
                shutil.copy(file, os.path.join(val_labels_dir, dir))
            
            # Copy desired files to test subfolder
            _file_operations.create_folder(os.path.join(test_labels_dir, dir))
            for file in validation_paths:
                shutil.copy(file, os.path.join(test_labels_dir, dir))

            # Checks that the number of files remain the same after the operation
            assert(len(training_paths) == len(os.listdir(os.path.join(train_labels_dir, dir))))
            assert(len(validation_paths) == len(os.listdir(os.path.join(val_labels_dir, dir))))  
            assert(len(validation_paths) == len(os.listdir(os.path.join(val_labels_dir, dir))))