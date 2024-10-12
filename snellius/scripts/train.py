
import utils
import pandas as pd
import os

test_set = [50, 100, 200, 300]
networks = ["unet", "deeplabv3", "attention_unet", "attention_resunet", "unet_pluplus"]

for i in range(len(test_set)):
    print(f"Testing {test_set[i]} epochs")
    for network in networks:
        execution_time = pd.Timestamp.now()
        TRAIN_CONFIG = {
            "training_path":"./data/final_images/train",
            "validation_path":"./data/final_images/validation", 
            "test_path":"./data/final_images/test",
            "network":network,
            "save_model":True,
            "downscaling_factor":32, #If specified you can keep track of performance in the excel file
            "n_classes":5,
            "batch_size": 16,
            "epochs": test_set[i],
            "validation_split":0.1,
            "test_split":0.1,
            "learning_rate":0.001,
            "early_stopping_patience":50,
            "lr_on_plateau_patience":49,
            "lr_mod_factor":0.1,
            "execution_time": execution_time.strftime("%Y%m%d_%H%M%S"),
            "results_folder": "results_" + execution_time.strftime("%Y%m%d_%H%M%S"),
            "verbose": 1 if os.name=="nt" else 0 # dissables verbose if we run on Snellius
        }
        utils.train_model(TRAIN_CONFIG)