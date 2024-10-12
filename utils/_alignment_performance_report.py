import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage.metrics import structural_similarity as ssim
from tqdm.notebook import tqdm_notebook

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2

MOCKUP_DICT = { 
"MSE": [],
"RMSE": [],
"Normalized RMSE": [],
"PSNR": [],
"SSIM": [],
"Jaccard Index": [],
"Cross Correlation": [],
"Cosine Similarity": [],
"Normalized TRE": [],
"Normalized ATRE": [],
"dA": []
}

def cosine_similarity(vec1, vec2):
    # We normalize the images to set their values between 0 and 1
    vec1 = vec1/255.0
    vec2 = vec2/255.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def mse_rmse_nrmse(img1, img2):
    mse = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    rmse = np.sqrt(mse)
    normalized_rmse = rmse / 255.0
    return mse, rmse, normalized_rmse

def psnr(mse):
    if mse == 0:
        psnr_result = float('inf')
    else:
        psnr_result = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr_result

def jaccard(mask1, mask2, binary_mask):
    if binary_mask:
        mask1 = mask1 // 255
        mask2 = mask2 // 255
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        jaccard_index = intersection / union if union != 0 else 1
    else:
        jaccard_index = None

    return jaccard_index

def tre_atre(img1, img2, landmarks1, landmarks2):
# TRE and ATRE (if landmarks are provided)
    if landmarks1 is not None and landmarks2 is not None:
        tre = np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1))
        atre = np.mean(tre)
        # Normalize TRE and ATRE
        norm_factor = np.linalg.norm([img1.shape[0], img1.shape[1]])
        normalized_tre = tre / norm_factor
        normalized_atre = atre / norm_factor
    else:
        normalized_tre = None
        normalized_atre = None

    return normalized_tre, normalized_atre

def da(mask1, mask2, binary_mask):
    if binary_mask:
        pre_area = np.sum(mask1)
        post_area = np.sum(mask2)
        delta_area = post_area - pre_area
    else:
        delta_area = None

    return delta_area

def append_results(results_dict, iter_dict):
    results_dict["MSE"].append(iter_dict["MSE"])
    results_dict["RMSE"].append(iter_dict["RMSE"])
    results_dict["Normalized RMSE"].append(iter_dict["Normalized RMSE"])
    results_dict["PSNR"].append(iter_dict["PSNR"])
    results_dict["SSIM"].append(iter_dict["SSIM"])
    results_dict["Jaccard Index"].append(iter_dict["Jaccard Index"])
    results_dict["Cross Correlation"].append(iter_dict["Cross Correlation"])
    results_dict["Cosine Similarity"].append(iter_dict["Cosine Similarity"])
    results_dict["Normalized TRE"].append(iter_dict["Normalized TRE"])
    results_dict["Normalized ATRE"].append(iter_dict["Normalized ATRE"])
    results_dict["dA"].append(iter_dict["dA"])

    return results_dict


def calculate_metrics(img1, img2, landmarks1=None, landmarks2=None, mask1=None, mask2=None, binary_mask=False):

    assert img1.shape == img2.shape, "Images must have the same dimensions"
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Mean Squared Error, Root Mean Squared Error, Normalised RMSE
    mse, rmse, norm_rmse = mse_rmse_nrmse(gray1, gray2)
    # Peak Signal-to-Noise Ratio (PSNR)
    psnr_result = psnr(mse)
    # Structural Similarity Index (SSIM)
    ssim_index, _ = ssim(gray1, gray2, full=True)
    # Jaccard Index
    jaccard_index = jaccard(gray1, gray2, binary_mask=False)
    # Cross Correlation
    cross_corr = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
    # Cosine Simmilarity
    cosine_sim = cosine_similarity(gray1.flatten(), gray2.flatten())
    # TRE and ATRE
    normalized_tre, normalized_atre = tre_atre(img1, img2, landmarks1, landmarks2)
    # Pre/Post Registration Change in Area (dA)
    delta_area = da(mask1, mask2, binary_mask)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "Normalized RMSE": norm_rmse,
        "PSNR": psnr_result,
        "SSIM": ssim_index,
        "Jaccard Index": jaccard_index,
        "Cross Correlation": cross_corr,
        "Cosine Similarity": cosine_sim,
        "Normalized TRE": normalized_tre,
        "Normalized ATRE": normalized_atre,
        "dA": delta_area
    }

def compute_results(results_dict):
    results_dict_copy = results_dict.copy()

    for x in results_dict_copy:
        if results_dict_copy[x][0] is not None:
            result_arr = np.array(results_dict_copy[x])
            if x == "Cosine Similarity":
                results_dict_copy[x] = str(np.round(np.mean(result_arr), 5)) + "+/-" + str(np.round(np.std(result_arr),5))
            else:   
                results_dict_copy[x] = str(np.round(np.mean(result_arr), 2)) + "+/-" + str(np.round(np.std(result_arr),2))
        else:
            results_dict_copy[x] = None

    return results_dict_copy

def measure_alignment(img_list, reshape=False):
    results_dict = { 
    "MSE": [],
    "RMSE": [],
    "Normalized RMSE": [],
    "PSNR": [],
    "SSIM": [],
    "Jaccard Index": [],
    "Cross Correlation": [],
    "Cosine Similarity": [],
    "Normalized TRE": [],
    "Normalized ATRE": [],
    "dA": []
    }

    i = 0
    while i < len(img_list)-1:
            if reshape:
                img1 = cv2.imread(img_list[i])
                height, width = img1.shape[0], img1.shape[1]
                img2 = cv2.imread(img_list[i+1])
                img2 = cv2.resize(img2, (width, height))
            else:
                img1 = cv2.imread(img_list[i])
                img2 = cv2.imread(img_list[i+1])
            #print("Comparing ", os.path.basename(img_list[i]), "VS", os.path.basename(img_list[i+1]))
            iter_dict = calculate_metrics(img1, img2, landmarks1=None, landmarks2=None, mask1=None, mask2=None, binary_mask=False)
            results_dict["MSE"].append(iter_dict["MSE"])
            results_dict["RMSE"].append(iter_dict["RMSE"])
            results_dict["Normalized RMSE"].append(iter_dict["Normalized RMSE"])
            results_dict["PSNR"].append(iter_dict["PSNR"])
            results_dict["SSIM"].append(iter_dict["SSIM"])
            results_dict["Jaccard Index"].append(iter_dict["Jaccard Index"])
            results_dict["Cross Correlation"].append(iter_dict["Cross Correlation"])
            results_dict["Cosine Similarity"].append(iter_dict["Cosine Similarity"])
            results_dict["Normalized TRE"].append(iter_dict["Normalized TRE"])
            results_dict["Normalized ATRE"].append(iter_dict["Normalized ATRE"])
            results_dict["dA"].append(iter_dict["dA"])
            i += 1
    
    final_dict = compute_results(results_dict)
    
    return final_dict

# We get the list of images from where we want to measure performance
# If images are not sorted use Natsort to sort them
def alignment_report(directory, resolution):
    print("Starting the image alignment process!!")
    df = pd.DataFrame(MOCKUP_DICT)
    
    processed_path = os.path.join(directory, "processed_"+str(resolution))

    unregistered_images = glob.glob(os.path.join(processed_path, "*.*"))
    preprocessed_unregistered_images = glob.glob(os.path.join(processed_path, "clean_dust_bubbles/*.*"))
    global_aligned_imgs = glob.glob(os.path.join(processed_path, "clean_dust_bubbles/registered/*.*"))
    elastic_aligned_imgs = glob.glob(os.path.join(processed_path, "clean_dust_bubbles/registered/elastic registration/*.*"))

    print("Elastic", len(elastic_aligned_imgs))
    print("Global", len(global_aligned_imgs))
    print("Unregistered", len(unregistered_images))
    print("Preprocessed Unregistered", len(preprocessed_unregistered_images))

    assert len(global_aligned_imgs) is not None and len(unregistered_images) is not None  and len(preprocessed_unregistered_images) is not None and len(elastic_aligned_imgs) is not None, "Check file paths"
    assert len(global_aligned_imgs) == len(elastic_aligned_imgs) == len(unregistered_images) == len(preprocessed_unregistered_images)

    result_dict = measure_alignment(unregistered_images, reshape=True)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Unregistered images done!")

    result_dict = measure_alignment(preprocessed_unregistered_images, reshape=True)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Preprocessed Images done!")

    result_dict = measure_alignment(global_aligned_imgs, reshape=False)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Global aligned Image done!")

    result_dict = measure_alignment(elastic_aligned_imgs, reshape=False)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Elastic aligned Image done!")

    print(df.head())

    return df

# We get the list of images from where we want to measure performance
# If images are not sorted use Natsort to sort them
def multiplex_alignment_report(directory, verbose=False):
    print("Measuring alignment metrics")
    df = pd.DataFrame(MOCKUP_DICT)

    unregistered_images = glob.glob(os.path.join(directory, "*.*"))
    global_aligned_imgs = glob.glob(os.path.join(directory, "registered/*.*"))
    elastic_aligned_imgs = glob.glob(os.path.join(directory, "registered/elastic registration/*.*"))

    print("Elastic", len(elastic_aligned_imgs))
    print("Global", len(global_aligned_imgs))
    print("Unregistered", len(unregistered_images))

    assert len(global_aligned_imgs) is not None and len(unregistered_images) is not None and len(elastic_aligned_imgs) is not None, "Check file paths"
    assert len(global_aligned_imgs) == len(elastic_aligned_imgs) == len(unregistered_images)

    result_dict = measure_alignment(unregistered_images, reshape=True)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Unregistered images done!")

    result_dict = measure_alignment(global_aligned_imgs, reshape=False)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Global aligned Image done!")

    result_dict = measure_alignment(elastic_aligned_imgs, reshape=False)
    temp_df = pd.DataFrame([result_dict])
    df = pd.concat([df, temp_df], ignore_index=True)
    print("Elastic aligned Image done!")
    if verbose:
        print(df.head())
    
    return df