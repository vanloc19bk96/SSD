import tensorflow as tf
from tensorflow.keras.backend import repeat
import numpy as np

def compute_iou(gt_boxes, prior_boxes):
    num_gt_boxes = len(gt_boxes)
    gt_boxes = repeat(tf.convert_to_tensor(gt_boxes, dtype=tf.float32), len(prior_boxes))
    prior_boxes = np.expand_dims(prior_boxes, 0)
    prior_boxes = tf.tile(prior_boxes, multiples=[num_gt_boxes, 1, 1])

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2], gt_boxes[..., 3]
    pri_xmin, pri_ymin, pri_xmax, pri_ymax = prior_boxes[..., 0], prior_boxes[..., 1], prior_boxes[..., 2], prior_boxes[..., 3]
    y0 = tf.maximum(gt_ymin, pri_ymin)
    y1 = tf.minimum(gt_ymax, pri_ymax)
    x0 = tf.maximum(gt_xmin, pri_xmin)
    x1 = tf.minimum(gt_xmax, pri_xmax)
    intersection = (x1 - x0) * (y1 - y0)
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pri_area = (pri_xmax - pri_xmin) * (pri_ymax - pri_ymin)
    union = gt_area + pri_area - intersection
    iou = intersection / union
    iou = np.array(iou)
    iou[iou < 0] = 0
    return tf.convert_to_tensor(iou)

def convert_center_to_corner(boxes):
    xmin = boxes[..., 0] - boxes[..., 2] / 2
    xmax = boxes[..., 0] + boxes[..., 2] / 2
    ymin = boxes[..., 1] - boxes[..., 3] / 2
    ymax  = boxes[..., 1] + boxes[..., 3] / 2
    corner_boxes = tf.transpose([xmin, ymin, xmax, ymax])
    return tf.cast(corner_boxes, tf.float32)

def convert_corner_to_center(boxes):
    cx = boxes[..., 0] + (boxes[..., 2] - boxes[..., 0]) / 2
    cy = boxes[..., 1] + (boxes[..., 3] - boxes[..., 1]) / 2
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    center_boxes = tf.transpose([cx, cy, w, h])
    return tf.cast(center_boxes, tf.float32)

def encode(gt_boxes, prior_boxes, variance):
    gt_boxes = tf.cast(gt_boxes, tf.float32)
    prior_boxes = tf.cast(prior_boxes, tf.float32)
    variance = tf.cast(variance, tf.float32)
    gx = (gt_boxes[..., 0] - prior_boxes[..., 0]) / (prior_boxes[..., 2] * variance[0])
    gy = (gt_boxes[..., 1] - prior_boxes[..., 1]) / (prior_boxes[..., 3] * variance[0])
    gw = tf.math.log(gt_boxes[..., 2] / prior_boxes[..., 2]) / variance[1]
    gh = tf.math.log(gt_boxes[..., 3] / prior_boxes[..., 3]) / variance[1]
    encoded_boxes = tf.transpose([gx, gy, gw, gh])
    return encoded_boxes
