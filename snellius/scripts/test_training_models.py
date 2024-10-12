import utils
import glob
import os

from patchify import patchify, unpatchify

import os
import glob
import cv2
import numpy as np
import glob
import natsort
import os

import utils._file_operations

from tensorflow.keras.models import load_model

def predict_image_stack_tiled(directory, model, folder, model_trained_size=128):
    """
    Gets all the images from a directory and predicts on
    them by creating patches of the model size
    """
    utils._file_operations.create_folder("./data/models_output_check")

    registered = glob.glob(directory)
    registered = natsort.natsorted(registered)

    for i, img_path in enumerate(registered):
        print(f"Predicting on image {img_path}")
        #Load the large RGB image
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
        #plt.imshow(reconstructed_image)

        img_fn = os.path.basename(img_path)
        img_fn_png = ".".join(img_fn.split(".")[:-1])+".png"

        print('Image write:',"Succedded" if cv2.imwrite(os.path.join("./data/models_output_check", folder + "_" + img_fn_png), reconstructed_image) else "Failed")

for folder in os.listdir("./trained_models"):
    #if "20240611" in folder:
    # Filter by date to avoid predicting on every model trained
    if "20240611" in folder:
        model = load_model(os.path.join("trained_models", folder, "unet.h5"), compile=False)
        print("Predicting on:", os.path.join("trained_models", folder, "unet.h5"))
        predict_image_stack_tiled(directory= "./data/66-4/processed_4/clean_dust_bubbles/registered/elastic registration/H21-066.4_HE332_077_Scan1.tif", model=model,folder=folder, model_trained_size=128)