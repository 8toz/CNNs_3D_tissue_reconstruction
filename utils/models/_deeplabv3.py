import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
# We can try out different pretrained models
from tensorflow.keras.applications import ResNet50

def ASPP(inputs, IMG_HEIGHT, IMG_WIDTH):
    shape = inputs.shape
    # Resnet factor
    factor = 0.0625
    y_pool = AveragePooling2D(pool_size=(int(IMG_HEIGHT*factor), int(IMG_WIDTH*factor)))(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation("relu")(y_pool)
    y_pool = UpSampling2D((int(IMG_HEIGHT*factor), int(IMG_WIDTH*factor)), interpolation="bilinear")(y_pool)
    # convolution 1x1
    y_1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation("relu")(y_1)
    # dilated x6
    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation("relu")(y_6)
    # dilated x12
    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation("relu")(y_12)
    # dilated x18
    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation("relu")(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def deepLabV3(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3):
    inputs = Input((None, None, IMG_CHANNELS))
    # Pretrained ResNet
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    #base_model.trainable = False

    image_features = base_model.get_layer("conv4_block6_out").output
    ## ASPP
    x_a = ASPP(image_features, IMG_HEIGHT, IMG_WIDTH)

    x_a = UpSampling2D((4,4), interpolation="bilinear")(x_a)

    x_b = base_model.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48, kernel_size=1, padding="same", use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    x = Concatenate()([x_a, x_b])

    x = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=256, kernel_size=1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D((4,4), interpolation="bilinear")(x)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax', name="output_layer")(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model