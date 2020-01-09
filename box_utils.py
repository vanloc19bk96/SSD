import tensorflow as tf
import numpy as np

def compute_iou(gt_box, prior_boxes):
    gt_box = tf.tile(gt_box, multiples=[len(prior_boxes), 1])
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box[:, 0], gt_box[:, 1], gt_box[:, 2], gt_box[:, 3]
    pri_xmin, pri_ymin, pri_xmax, pri_ymax = prior_boxes[:, 0], prior_boxes[:, 1], prior_boxes[:, 2], prior_boxes[:, 3]

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

# zeros = np.zeros((300, 300))
# gt_box = np.array([[1, 2, 90, 250]])
# cv2.rectangle(zeros, (gt_box[0][0], gt_box[0][1]), (gt_box[0][2], gt_box[0][3]), (255, 255, 255), 2)
# prior_boxes = np.array([[1, 2, 90, 250]])
# cv2.rectangle(zeros, (prior_boxes[0][0], prior_boxes[0][1]), (prior_boxes[0][2], prior_boxes[0][3]), (255, 255, 255), 2)
# cv2.imshow("image", zeros)
# cv2.waitKey(0)
# print(compute_iou(gt_box, prior_boxes))