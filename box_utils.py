import tensorflow as tf
from tensorflow.keras.backend import repeat
import numpy as np

def compute_iou(gt_boxes, prior_boxes):
    num_gt_boxes = len(gt_boxes)
    gt_boxes = repeat(gt_boxes, len(prior_boxes))
    prior_boxes = np.expand_dims(prior_boxes, 0)
    prior_boxes = tf.tile(prior_boxes, multiples=[num_gt_boxes, 1, 1])

    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_boxes[..., 0], gt_boxes[..., 1], gt_boxes[..., 2], gt_boxes[..., 3]
    pri_xmin, pri_ymin, pri_xmax, pri_ymax = prior_boxes[..., 0], prior_boxes[..., 1], prior_boxes[..., 2], prior_boxes[..., 3]
    y0 = np.maximum(gt_ymin, pri_ymin)
    y1 = np.minimum(gt_ymax, pri_ymax)
    x0 = np.maximum(gt_xmin, pri_xmin)
    x1 = np.minimum(gt_xmax, pri_xmax)
    intersection = (x1 - x0) * (y1 - y0)
    gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
    pri_area = (pri_xmax - pri_xmin) * (pri_ymax - pri_ymin)
    union = gt_area + pri_area - intersection
    iou = intersection / union

    return iou

# import cv2
#
# zeros = np.zeros((300, 300))
# gt_boxes = tf.constant([[1, 2, 90, 250], [1, 2, 4, 9]])
# cv2.rectangle(zeros, (gt_boxes[0][0], gt_boxes[0][1]), (gt_boxes[0][2], gt_boxes[0][3]), (255, 255, 255), 2)
# prior_boxes = np.array([[1, 2, 60, 100], [3, 4, 5, 6], [6, 7, 8, 9], [9, 10, 11, 12]])
#
# cv2.rectangle(zeros, (prior_boxes[0][0], prior_boxes[0][1]), (prior_boxes[0][2], prior_boxes[0][3]), (255, 137, 226), 2)
# cv2.imshow("image", zeros)
# cv2.waitKey(0)
# print(compute_iou(gt_boxes, prior_boxes))