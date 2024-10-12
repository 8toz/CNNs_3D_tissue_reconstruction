import utils
import utils._file_operations

import os
import glob
import cv2
import re
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import keras
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from keras.metrics import MeanIoU

import keras.backend as K
import tensorflow as tf

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.0 * intersection + 1.0)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (intersection + 1.0)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def prepare_image_set(directory):
    """
    Given a directory preprocess the images and returns an X, y set
    """
    #Capture training image info as a list
    train_images = []

    for img_path in glob.glob(os.path.join(directory, "images", "*.png")):
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)       
        train_images.append(img)
        
    #Convert list to array for machine learning processing        
    train_images = np.array(train_images)

    #Capture mask/label info as a list
    train_masks = [] 

    for mask_path in glob.glob(os.path.join(directory, "masks", "*.png")):
        mask = cv2.imread(mask_path, 0)       
        train_masks.append(mask)
       

    train_masks = np.array(train_masks)
    # Normalize images
    train_images = train_images/255.0

    return train_images, train_masks


def train_model(TRAIN_CONFIG):

    utils._file_operations.create_folder(os.path.join("./trained_models", TRAIN_CONFIG["results_folder"]))

    
    X_train, y_train = prepare_image_set(TRAIN_CONFIG["training_path"])
    X_val, y_val = prepare_image_set(TRAIN_CONFIG["validation_path"])
    X_test, y_test = prepare_image_set(TRAIN_CONFIG["test_path"])

    print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background 

    # Convert masks to categorical (should have shape (height, width, n_classes))
    y_train_cat = to_categorical(y_train, num_classes=TRAIN_CONFIG["n_classes"])
    y_val_cat = to_categorical(y_val, num_classes=TRAIN_CONFIG["n_classes"])
    y_test_cat = to_categorical(y_test, num_classes=TRAIN_CONFIG["n_classes"])


    # Compute class weights (not used atm going for image augmentation)
    # class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(train_masks),y=train_masks.flatten())
    # class_weights_dict = dict(enumerate(class_weights))
    # class_weights_dict


    # Check that the shape of all images is the same
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]


    def get_model(network):
        if network == "unet":
            print("Loading first UNET model")
            return utils.unet_model(n_classes=TRAIN_CONFIG["n_classes"], IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
        elif network == "deeplabv3":
            print("Loading DEEPLAB V3 model")
            return utils.deepLabV3(n_classes=TRAIN_CONFIG["n_classes"], IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
        elif network == "attention_unet":
            print("Loading Attention UNET model")
            return utils.attention_unet(n_classes=TRAIN_CONFIG["n_classes"], dropout_rate=0.2, batch_norm=True)
        elif network == "attention_resunet":
            print("Loading Attention ResUnet")
            return utils.attention_resunet(n_classes=TRAIN_CONFIG["n_classes"], dropout_rate=0.2, batch_norm=True)
        elif network == "unet_pluplus":
            print("Loading Unet++")
            return utils.unet_plusplus(n_classes=TRAIN_CONFIG["n_classes"])
        else:
            raise "Please load a valid network: unet, deeplabv3, attention_unet, attention_resunet, unet_pluplus"

    keras.utils.set_random_seed(32)
    model = get_model(network = TRAIN_CONFIG["network"])
    optimizer = keras.optimizers.Adam(learning_rate=TRAIN_CONFIG["learning_rate"])
    callback_ES = keras.callbacks.EarlyStopping(monitor='val_loss', patience=TRAIN_CONFIG["early_stopping_patience"], restore_best_weights=True)
    callback_LR = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=TRAIN_CONFIG["lr_mod_factor"], patience=TRAIN_CONFIG["lr_on_plateau_patience"])

    model.compile(optimizer=optimizer, loss=categorical_focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy', jacard_coef])
    #model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy', jacard_coef])

    model.summary()


    #If starting with pre-trained weights. 
    #model.load_weights('???.hdf5')

    start = datetime.now()
    history = model.fit(X_train, y_train_cat, 
                        batch_size = TRAIN_CONFIG["batch_size"], 
                        verbose=TRAIN_CONFIG["verbose"], 
                        epochs=TRAIN_CONFIG["epochs"], 
                        validation_data=(X_val, y_val_cat), 
                        callbacks=[callback_ES, callback_LR],
                        # sample_weight=class_weights,
                        shuffle=False)
    stop = datetime.now()
    execution_time = stop - start
    print(f"The Neural Network training took {execution_time}")


    # Model evaluation
    _, acc, jaq = model.evaluate(X_test, y_test_cat, batch_size=8)
    print("Accuracy is = ", (acc * 100.0), "%")


    y_pred = model.predict(X_test, batch_size=6)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    # y_pred_argmax_unstacked = utils.unstack_labels(y_pred_argmax)
    # y_test_unstacked = utils.unstack_labels(y_test)


    conf_matrix = confusion_matrix(y_test.reshape(-1), y_pred_argmax.reshape(-1))
    clf_report = classification_report(y_test.reshape(-1), y_pred_argmax.reshape(-1), output_dict=True)


    print(classification_report(y_test.reshape(-1), y_pred_argmax.reshape(-1)))


    IOU_keras = MeanIoU(num_classes=TRAIN_CONFIG["n_classes"])  
    IOU_keras.update_state(y_test, y_pred_argmax)
    mean_IoU = IOU_keras.result().numpy()
    print("Mean IoU =", mean_IoU)


    values = np.array(IOU_keras.get_weights()).reshape(TRAIN_CONFIG["n_classes"], TRAIN_CONFIG["n_classes"])

    class0_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0] + values[2,0]+ values[3,0] + values[4,0]) # R
    class1_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1] + values[2,1]+ values[3,1] + values[4,1])
    class2_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4] + values[0,2] + values[1,2]+ values[3,2] + values[4,2])
    class3_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3] + values[1,3]+ values[2,3] + values[4,3])
    class4_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[4,3] + values[3,4] + values[1,4] + values[2,4] + values[0, 4])

    print("IoU for BACKGROUND is: ", class0_IoU)
    print("IoU for EPITHELIUM is: ", class1_IoU)
    print("IoU for BLOOD VESSELS is: ", class2_IoU)
    print("IoU for STROMA is: ", class3_IoU)
    print("IoU for ADIPOCYTES is: ", class4_IoU)


    iou_results = {
        "mean_iou": mean_IoU,
        "epithelium_iou": class1_IoU,
        "blood_vessel_iou":class2_IoU,
        "stroma_iou": class3_IoU,
        "adipocytes_iou": class4_IoU,
    }

    utils.results_to_excel(TRAIN_CONFIG, accuracy=acc, clf_report=clf_report, iou_results=iou_results)

    # Save confussion matrix
    np.save(os.path.join("./trained_models/", TRAIN_CONFIG["results_folder"], "conf_matrix.npy"), conf_matrix)

    # Save accuracy results as pickle file
    with open(os.path.join("./trained_models/", TRAIN_CONFIG["results_folder"], "clf_results.pkl"), 'wb') as f:
        pickle.dump(clf_report, f)

    # Writes the history and model
    with open(os.path.join("./trained_models/", TRAIN_CONFIG["results_folder"], "history.pkl"), 'wb') as f:
        pickle.dump(history.history, f)

    if TRAIN_CONFIG["save_model"]:
        model.save(os.path.join("./trained_models/", TRAIN_CONFIG["results_folder"], "unet.h5"))
