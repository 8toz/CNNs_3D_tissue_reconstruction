import utils

import napari
# Uncomment when using dask 
# import dask
# import dask.array as da
import numpy as np
import os
import glob

from natsort import natsorted
from napari_animation import Animation

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2

def generate_render(imgs_path, predictions_path, record_video=False, format="2D"):

    # Store images
    he_file_names = glob.glob(imgs_path)
    he_file_names = natsorted(he_file_names)

    predictions = glob.glob(predictions_path)
    predictions = natsorted(predictions)

    predictions_stack = []

    he_img = []
    for _file in he_file_names:
        img = cv2.imread(_file)
        # IMPORTANT
        # In order to match the shape of the predictions we must 
        # pad the image the same way as it is done in the predictions
        pad_image = utils._image_operations.white_pad_image(img, 128)
        he_img.append(pad_image)

    he_img = np.array(he_img)
    
    for _file in predictions:
        img = cv2.imread(_file, 0)
        # Change color of labels
        img = np.where(img == 2, 10, img)
        img = np.where(img == 1, 12, img)
        img = np.where(img == 3, 16, img)
        img = np.where(img == 4, 9, img)

        predictions_stack.append(img.astype("int64"))

    predictions_stack = np.array(predictions_stack)

    # We check the shape to see if the padding applied to predictions is 
    # The same as for the H&E
    assert predictions_stack.shape == he_img.shape[:-1]

    # Initialize the viewer
    viewer = napari.Viewer()
    viewer.add_image(he_img, scale=(20,4,4))
    viewer.add_labels(predictions_stack, scale=(20,4,4))

    # TODO activate labels with a nice opacity setting to get a smooth video
    # TODO fine tune the parameters to get a better video, now it goes a bit fast and not fluid

    # Store predictions
    if record_video:
        if format == "2D":
            twoD_path = "./videos/animate2D_32.mp4"
            print("Generating 2D video...")
            animation = Animation(viewer)
            viewer.update_console({"animation": animation})
            viewer.layers[-1].visible = False
            viewer.dims.current_step = (0, 0, 0)
            animation.capture_keyframe(steps=1)
            viewer.dims.current_step = (100, 0, 0)
            animation.capture_keyframe(steps=120)
            animation.animate(twoD_path, canvas_only=True)
            print(f"Video successfully saved at {twoD_path}")


        elif format == "3D":
            threeD_path = './videos/test3D_32.mov'
            print("Generating 3D video...")
            animation = Animation(viewer)
            viewer.layers[-1].visible = True
            viewer.layers[0].visible = False
            viewer.dims.ndisplay = 3
            viewer.layers[0].rendering = "translucent"
            viewer.camera.angles = (0.0, 0.0, 0.0)
            animation.capture_keyframe()
            viewer.camera.angles = (0.0, 180.0, 0.0)
            animation.capture_keyframe(steps=120)
            viewer.camera.angles = (0.0, 360.0, 0.0)
            animation.capture_keyframe(steps=120)
            animation.animate(threeD_path, canvas_only=True)
            f"Video successfully saved at {threeD_path}"

    
