import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import tensorflow as tf

def create_seq(image, bbs, labels):
    seq = iaa.Sequential([
        iaa.Resize({"height": 300, "width": 300}),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=[0, 1],
            cval=(0, 255),
            mode=ia.ALL
        )
    ])
    image = np.array(image)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug, labels

def augment_gt_boxes(image, gt_boxes):
    bounding_boxes = []
    for gt_box in gt_boxes:
        bounding_boxes.append(BoundingBox(x1=gt_box[0], y1=gt_box[1], x2=gt_box[2], y2=gt_box[3]))
    bbs = BoundingBoxesOnImage(bounding_boxes, shape=tuple(image.shape))
    return bbs

def augment(image, gt_boxes, labels):
    bbs = augment_gt_boxes(image, gt_boxes)
    seq = create_seq(image, bbs, labels)
    image_aug, bbs_aug, labels = seq
    return tf.convert_to_tensor(image_aug, dtype=tf.float32), tf.convert_to_tensor(format_gt_boxes(bbs_aug), dtype=tf.float32), tf.convert_to_tensor(labels, dtype=tf.float32)

def format_gt_boxes(bbs_aug):
    bounding_boxes = []
    for i in range(len(bbs_aug.bounding_boxes)):
        bounding_box = bbs_aug.bounding_boxes[i]
        bounding_boxes.append([bounding_box.x1, bounding_box.y1, bounding_box.x2, bounding_box.y2])

    return bounding_boxes