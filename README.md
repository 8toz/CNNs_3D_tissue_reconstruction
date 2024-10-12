# Deep Learning for 3D Tissue reconstruction

<div align="center">
  <img src="./gifs/3D%20View.gif" alt="HE_dark_bg_predictions" style="display: block; margin: 0 auto; width: 500px; height: auto;" />
</div>

## Table of Contents

- [Abstract](#abstract)
- [Installation](#installation)
- [Notebooks](#notebooks)
- [Presentation](#presentation)

## Abstract 
In this project we present an automated pipeline for the 3D reconstruction
of breast tissue images and the determination of their
cellular heterogeneity by combining H&E and multiplex images.
The pipeline consists of three major tasks: image registration, tissue
segmentation through Deep Learning, and the final 3D volume
reconstruction. We `proved the feasibility of combining two distinct
image types (H&E with multiplex) into an unique 3D model`, allowing
the inspection of the breast tissue with a new unexplored
perspective.

In the image alignment stage, we employ a combination of global
and local transformations to accurately reorient the serial sections.
For tissue classification, a `U-Net architecture` is trained by using
manually labeled H&E images, achieving an average `Jaccard score
of 96.79%`. The final 3D reconstruction is created by integrating
the aligned images and the classification results, enabling the exploration
of the breast tissue architecture interactively up to `1 𝜇𝑚
resolutions`.

The results show that the proposed pipeline effectively reconstructs
3D volumes from 2D images. It achieves State of the Art
accuracies in segmentation performance and is four times faster
than current solutions, owing to its new Python implementation.
This method has the potential to enhance our understanding of
breast tissue anatomy and physiology, and to serve as a valuable
tool for future research in microscope imaging.

## Installation 

1. A system with a GPU installed and CUDA support is required for optimal performance, as the program utilizes TensorFlow.
2. This installation guide is intended for Linux distributions.
3. Matlab is needed for the image registration step.

### Steps
1. Ensure you have CUDA properly installed. You can follow the [official CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for Linux.
2. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/8toz/CNNs_3D_tissue_reconstruction.git
   ```
3. Navigate to the project directory
   ```bash
   cd <PROJECT_DIRECTORY_NAME>
   ```
4. Install the requirements file
    ```bash
   pip install -r requirements.txt
   ```
5. For the image registration clone the CODA repo and place the files in `./image_registration_scripts` path. 
   ```bash
   git clone https://github.com/ashleylk/CODA.git
   ```
   Here you can find more information about [CODA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10500590/). Amazing work!!
## Notebooks

In this notebooks you'll find an overview of the main functions available in the repository. We have organized the content into four Jupyter notebooks, each with a distinct purpose:

### 1. **HE_Pipeline.ipynb**
   - This notebook handles the initial steps of the pipeline, which involve downscaling and registering the images using CODA.
   - If you don't have a trained model or wish to train a new one, refer to `train_HE.ipynb`.
   - Once the model is ready, the remaining steps will use the trained model to predict and reconstruct the 3D tissue using Napari.

### 2. **train_HE.ipynb**
   - This notebook is responsible for training the models. You will need labels and the downscaled images from the previous step.
   - The pipeline will build the mosaic, create training datasets, and train various models.
   - After training, you can evaluate the model performance by using the `visualize_train_results.ipynb`.

### 3. **multiplex_pipeline.ipynb**
   - This notebook performs similar tasks to `HE_Pipeline.ipynb` but for multiplex images.
   - Please note that this block is still under development and lacks proper commenting, making the code a bit messy.
   - The necessary MATLAB files are found in the `image_registration_scripts` directory (`0_main.m`). These files will help process the multiplex data and generate final images, similar to the one on the cover of the repository.

### 4. **visualize_train_results.ipynb**
   - This notebook allows you to visualize the results from the training phase, showing metrics and other performance indicators for the trained models.

## Presentation 
[Power Point URL](./3D%20Breast%20Reconstruction%20with%20Deep%20Learning.pptx)
