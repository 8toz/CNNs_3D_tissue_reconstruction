import os

# Pyvips is not installed in Snellius
if os.name == 'nt':
    vipsbin = r"c:\vips-dev-8.15\bin"
    os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']
    import pyvips

from tqdm.notebook import tqdm_notebook
from tifffile import TiffFile
import os
from xml.etree import ElementTree

### MAIN METHOD ###
def resize_qptiff_files(folder, target_size): 
    """
    Gets the qptiff files from a folder and process them to a specific target size.
    As off now the method only extracts the resolutions from the piramidal image
        Original, x2, x4, x8, x16, x32
    """
    files = get_qptiff_paths(folder)
    for file in tqdm_notebook(files):
        process_qptiff_image(file, folder, target_size)

    return "Images processed and saved"

### Auxiliar methods called within the main one ###

def process_qptiff_image(img_path, folder, desired_file_size):
    """
    For each qptiff image found downscales it to the desired resolution
    taking care if its HE or multiplex in the background.
        - For an HE we get 1 image per qptiff file 
        - For multiplex we get 8 images corresponding to each of
        the multiplexed channels
    """
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    with TiffFile(img_path) as tif:
        # Detects original size on the piramidal image
        max_height, max_width = get_max_size(img_path)
        if is_multiplex(img_path):
            channels = get_channel_names(img_path)
        counter = 0

        for i, page in enumerate(tif.pages):
            pages = str(tif.pages[i])
            # Get the resolution for each image
            height, width = get_resolution(pages)
            page_dict = create_page_dict(pages, max_height, max_width, height, width)
            if (str(page_dict["downsampling_factor"]) == desired_file_size) and (page_dict["is_reduced"] or page_dict["downsampling_factor"]=="original"):
                create_folder(os.path.join(folder,"processed"+ "_" +desired_file_size))
                array = page.asarray()
                image = pyvips.Image.new_from_array(array)
                if is_multiplex(img_path):
                    image.write_to_file(os.path.join(folder,
                                                 "processed"+ "_" +desired_file_size,
                                                 file_name +"_"+channels[counter]+".tif")
                                    , pyramid=True
                                    , tile=True)
                    counter += 1
                else:
                    image.write_to_file(os.path.join(folder,
                                                 "processed"+ "_" +desired_file_size,
                                                 file_name+".tif")
                                    , pyramid=True
                                    , tile=True)
                    
def create_folder(folder_path):
    '''
    Creates a folder in the requested path if it does not already exists
    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass

def get_qptiff_paths(path):
    """
    Given a path gets all the qptiff files inside it and return a list 
    with its adresses
    """
    path_list = []
    for paths in os.walk(path, topdown=False):
        root = paths[0]
        for file in paths[-1]:
            if file.split(".")[-1] == "qptiff":
                path_list.append(os.path.join(root, file))

    return path_list

def get_resolution(input_str):
    """
    Given a string gets its resolution (This works based on image metadata)
    At the 4th position we have HEIGTH x WIDTH
    """
    input_str = str(input_str) # Make sure is str format
    size = input_str.split(" ")[4]
    height, width = size.split("x")[0], size.split("x")[1]
    return int(height), int(width)

def get_max_size(path):
    """
    By using the prev. method get_resolution checks every page of the
    image metadata and returns the max value (which correspon to the
    original image size)
    """
    with TiffFile(path) as tif:
        height, width = -1, -1
        for i, page in enumerate(tif.pages):
            pages = str(tif.pages[i])
            img_height, img_width = get_resolution(pages)
            height = img_height if height < img_height else height
            width = img_width if width < img_width else width
    
    return height, width

def create_page_dict(page, max_height, max_width, height, width):
    """
    Given a page creates a dictionary storing its information.
    Checks the rgb and reduced features.
    The Downsampling factor returns Original or its smaller sizes.
    This method will serve to get the desired PYRAMIDAL PAGE information
    """
    is_rgb = True if "rgb" in page else False
    is_reduced = True if "reduced" in page else False
    downsampling_factor = "original" if int(max_height/height) == 1 else int(max_height/height)
    page_split = str(page).split(" ")
    result = {
        "page_number":page_split[1],
        "memory_position":page_split[2],
        "image_size": page_split[4],
        "is_rgb": is_rgb,
        "is_reduced": is_reduced,
        "downsampling_factor": downsampling_factor
    }    
    return result
    

def is_multiplex(path):
    """
    Checks if the image is a multiplex one. 
    The rule is very stupid maybe it can be improved or simplified.
    If the number of pages is large is a multiplexed one, if not HE.
    """
    with TiffFile(path) as tif:
        if len(tif.pages) < 30:
            return False
    return True

def check_qptiff_data(path):
    """
    Given an image path check the content of an image 
    (its metadata in pages format)
    """
    with TiffFile(path) as tif:
        for i, page in enumerate(tif.pages):
            print(page)
    return "Done"

def get_channel_names(img_path):
    """
    For the multiplex case we want to get the channel names 
    to know which fluorophore was used.
    """
    channels = []
    with TiffFile(img_path) as tif:
        for page in tif.series[0].pages:
            channel = ElementTree.fromstring(page.description).find('Name').text
            channel = channel.replace(" ", "_")
            channels.append(channel)
    return channels

