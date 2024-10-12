import utils

from tensorflow.keras.models import load_model

# Load the model
model = load_model('unet.h5', compile=False)

#utils.resize_qptiff_files("./data/66-4", "4")

#utils.remove_dustNbubbles("./data/66-4/processed_4")

# TODO EXECUTE MATLAB ALIGMENT process

# Check if background is classified correctly if not cleaned predictions will most likely fail
utils.predict_stack_untiled("./data/66-4/processed_4/clean_dust_bubbles/registered/elastic registration/*.jpg", model, model_trained_size=256)
