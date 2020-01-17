from tensorflow.keras.layers import Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Model

def vgg_block(input_x, repetition, filters, final):
    x = input_x
    for i in range(repetition):
        x = Conv2D(filters=filters, kernel_size=(3, 3), padding="same", activation="relu")(x)
    if not final:
        x = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(x)

    return x

def VGG16(input_x):
    x = vgg_block(input_x, 2, 64, False)
    x = vgg_block(x, 2, 128, False)
    x = vgg_block(x, 3, 256, False)
    vgg_conv4 = vgg_block(x, 3, 512, True)
    x = vgg_block(x, 3, 512, False)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(x)
    vgg_conv7 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", name="conv7")(x)
    return vgg_conv4, vgg_conv7

# import cv2
# import tensorflow as tf
#
# img = cv2.imread(r"C:\Users\Admin\Desktop\1.jpg")
# img = cv2.resize(img, (300, 300))
# input_x = tf.cast(tf.expand_dims(img, 0), tf.float32)
# VGG16()[2](input_x)