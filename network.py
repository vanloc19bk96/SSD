from vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Input
import tensorflow as tf


NUM_BOXES_PER_FEATURE = [4, 6, 6, 6, 4, 4]
def extra_feature_layers(vgg_conv4, vgg_conv7, classes):
    classifiers = []

    classifier_1 = Conv2D(4*(classes + 5), kernel_size=(3, 3), padding="same")(vgg_conv4)
    classifier_2 = Conv2D(filters=6*(classes + 5), kernel_size=(3, 3), padding="same")(vgg_conv7)
    classifiers.append(classifier_1)
    classifiers.append(classifier_2)

    conv8 = Conv2D(filters=256, kernel_size=(1, 1), padding="same")(vgg_conv7)
    conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding="same")(conv8)
    classifier_3 = Conv2D(filters=6*(classes + 5), kernel_size=(3, 3), padding="same")(conv8)
    classifiers.append(classifier_3)

    conv9 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(conv8)
    conv9 = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")(conv9)
    classifier_4 = Conv2D(filters=6*(classes + 5), kernel_size=(3, 3), padding="same")(conv9)
    classifiers.append(classifier_4)

    conv10 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(conv9)
    conv10 = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding="same")(conv10)
    classifier_5 = Conv2D(filters=4*(classes + 5), kernel_size=(3, 3), padding="same")(conv10)
    classifiers.append(classifier_5)

    conv11 = Conv2D(filters=128, kernel_size=(1, 1), padding="same")(conv10)
    conv11 = Conv2D(filters=256, kernel_size=(3, 3), strides=1)(conv11)
    classifier_6 = Conv2D(filters=4*(classes + 5), kernel_size=(3, 3), padding="same")(conv11)
    classifiers.append(classifier_6)
    return classifiers

def compute_output(classifiers):
    output = []
    for index, classifier in enumerate(classifiers):
        classifier = tf.expand_dims(classifier, -1)
        batch, h, w = classifiers[0].shape[:3]
        _classifier = tf.reshape(classifier, [batch, h, w, NUM_BOXES_PER_FEATURE[index], -1])
        output.append(_classifier)
    return output

def ssd(classes):
    input_x = Input(shape=(300, 300, 3))
    vgg_conv4, vgg_conv7 = VGG16(input_x)
    classifiers = extra_feature_layers(vgg_conv4, vgg_conv7, classes)
    output = compute_output(classifiers)
    return output
