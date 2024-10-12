import utils._config as _config

import pandas as pd
import numpy as np
import os
import glob
import solt
import random
from solt import transforms as slt

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2

def augment_labels(directory=_config.LABELS_SOURCE_PATH):

    df_labels = check_label_balance(labels_path=directory)
    print(df_labels)
    max_pixel_goal = df_labels["n_pixels"].max()

    for idx, row in df_labels.iterrows():
        print(idx)
        current_pixels = row["n_pixels"]
        counter = 0
        while current_pixels < max_pixel_goal:
            labels_path = os.path.join(directory, idx)
            # select random image from list 
            random_image = random.choice(os.listdir(labels_path))

            augmented_img = random_augmentation(os.path.join(directory, idx, random_image), p=0.5)
            
            cv2.imwrite(os.path.join(labels_path, "augmented_"+str(idx)+"_"+str(counter))+".png", augmented_img)

            # Count pixels
            gray_image = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
            pixels = np.sum(np.where(gray_image > 0, 1, 0))
            
            current_pixels += pixels
            counter += 1


def count_pixels(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pixels = np.sum(np.where(img > 0, 1, 0))
    return pixels

def check_label_balance(labels_path=_config.LABELS_SOURCE_PATH):
    labels_dict = {}
    count_dict = {}
    for dir in os.listdir(labels_path):
        count_dict[dir] = 0
        files = glob.glob(os.path.join(labels_path, dir, "*.png"))
        labels_dict[dir] = len(files)
        for file_path in files:
            count_dict[dir] += count_pixels(file_path)

    result = pd.DataFrame(pd.Series(count_dict))
    result["n_labels"] = pd.Series(labels_dict)
    result["relative_percentage"] = result[0]/result[0].max()
    result["absolute_percentage"] = result[0]/result[0].sum()
    result.rename(columns={0:"n_pixels"}, inplace=True)

    return result[["n_labels", "n_pixels", "absolute_percentage", "relative_percentage"]]

def random_augmentation(path, p: float) -> None:
    """

    """
    img = cv2.imread(path)

    stream = solt.Stream([
        #slt.Rotate(angle_range=(-45, 45), p=p, padding='r'),
        slt.Rotate(angle_range=(-45, 45), p=p),
        slt.Flip(axis=1, p=p/2),
        slt.Flip(axis=0, p=p/2),
        #slt.Scale(range_x=(0.8, 1.2), padding='r', range_y=(0.8, 1.2), same=False, p=p),
        slt.Scale(range_x=(0.7, 1.2), range_y=(0.7, 1.2), same=False, p=p),
        #slt.Blur(k_size=7, blur_type='m', p=p / 2),
        ], ignore_fast_mode=True)

    aug_img = stream({"image": img}, return_torch=False).data[0]
    
    return aug_img

