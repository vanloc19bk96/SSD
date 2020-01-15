from tensorflow.keras.layers import Input, Conv2D, MaxPool2D


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
    x = vgg_block(x, 3, 512, False)
    vgg_conv4 = vgg_block(x, 3, 512, True)
    # vgg_conv4 = Model(input_x, out)

    # input_x = Input(shape=[19, 19, 512])
    x = MaxPool2D(pool_size=(3, 3), strides=1, padding="same")(vgg_conv4)
    x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(x)
    vgg_conv7 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", name="conv7")(x)
    # vgg_conv7 = Model(input_x, out)

    return vgg_conv4, vgg_conv7

# input = Input(shape=[300, 300, 3])
# extra_feature_layers(input, 3)