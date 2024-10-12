import utils
import utils._image_operations
from utils._smooth_predictions import predict_img_with_smooth_windowing

import os
import glob
import cv2
import numpy as np
import glob
import natsort
import os

import gc
import keras
import keras.backend as K

from patchify import patchify, unpatchify

def predict_image_stack_tiled(directory, model, model_trained_size=256):
    """
    Gets all the images from a directory and predicts on
    them by creating patches of the model size
    """
    registered = glob.glob(directory)
    registered = natsort.natsorted(registered)

    for i, img_path in enumerate(registered):
        print(f"Predicting on image {img_path}")
        # Load the large RGB image
        large_image = cv2.imread(img_path)

        pad_image = utils._image_operations.white_pad_image(large_image, model_trained_size)

        # Split the image into patches
        patches = patchify(pad_image, (model_trained_size, model_trained_size, 3), step=model_trained_size)

        predicted_patches = []

        # Loop through each patch
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                # Get a single patch
                single_patch = patches[i, j, :, :, :]

                # Normalize the patch
                single_patch_norm = single_patch / 255.0
                #print(single_patch_norm.shape)
                # Predict the patch
                single_patch_prediction = model.predict(single_patch_norm, verbose=0)

                # Get the predicted segmentation mask
                single_patch_predicted_img = np.argmax(single_patch_prediction, axis=-1)[0, :, :]

                predicted_patches.append(single_patch_predicted_img)

        # Reshape the predicted patches
        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], model_trained_size, model_trained_size))

        # Reconstruct the image from predicted patches
        reconstructed_image = unpatchify(predicted_patches_reshaped, pad_image.shape[:2])

        img_fn = img_path.split("\\")[-1]
        img_fn_png = ".".join(img_fn.split(".")[:-1])+".png"

        print('Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions", img_fn_png), reconstructed_image) else "Failed")

        cleaned_image = clean_predictions(reconstructed_image.astype("uint8"))

        print('Clean Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions_cleaned", img_fn_png), cleaned_image) else "Failed")

def predict_stack_untiled(directory, model, model_trained_size=128):
    registered = glob.glob(directory)
    registered = natsort.natsorted(registered)

    for i, img_path in enumerate(registered):
        print(f"Predicting on image {img_path}")
        # Load the large RGB image
        large_image = cv2.imread(img_path)
        pad_image = utils._image_operations.white_pad_image(large_image, model_trained_size)
        prediction_proba = model.predict(np.expand_dims(pad_image/255.0, axis=0))
        K.clear_session()
        gc.collect()
        prediction_img = np.argmax(prediction_proba[0], axis=-1)


        img_fn = img_path.split("\\")[-1]
        img_fn_png = ".".join(img_fn.split(".")[:-1])+".png"

        print('Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions", img_fn_png), prediction_img) else "Failed")

        cleaned_image = clean_predictions(prediction_img.astype("uint8"))

        print('Clean Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions_cleaned", img_fn_png), cleaned_image) else "Failed")


def predict_image_stack_tiled_smoothed(directory, model, model_trained_size=128, n_classes=5):
    registered = glob.glob(directory)
    registered = natsort.natsorted(registered)

    for i, img_path in enumerate(registered):
        large_image = cv2.imread(img_path)
        print(f"Predicting on image {img_path}")
        pad_image = utils._image_operations.white_pad_image(large_image, model_trained_size)
        # Normalize image
        input_img = pad_image / 255.0

        # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, 
        # called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=model_trained_size,
            subdivisions=2, 
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict((img_batch_subdiv), verbose=0)
            )
        )
        
        prediction_img = np.argmax(predictions_smooth, axis=-1)


        img_fn = img_path.split("\\")[-1]
        img_fn_png = ".".join(img_fn.split(".")[:-1])+".png"

        print('Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions", img_fn_png), prediction_img) else "Failed")

        cleaned_image = clean_predictions(prediction_img.astype("uint8"))

        print('Clean Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/predictions_cleaned", img_fn_png), cleaned_image) else "Failed")

        K.clear_session()
        gc.collect()


def clean_predictions(img):
    """
    Given a prediction removes background noise by applying contour detection
    returns the cleaned prediction for the 3D model
    """

    bck_img = img.copy()
    contours, _ = cv2.findContours(bck_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(bck_img)

    # Find largest contour
    max_area = -1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    mask = cv2.drawContours(contour_img, [max_contour], -1, (255), thickness=cv2.FILLED)
    cleaned_img = np.where(mask==255, img, 0)

    return cleaned_img