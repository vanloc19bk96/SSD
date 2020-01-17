from box_utils import *
from prior_generation import *

def compute_target(gt_boxes, prior_boxes, gt_labels, threshold = .5):
    transformed_prior_boxes = convert_center_to_corner(prior_boxes)
    ious = compute_iou(gt_boxes, transformed_prior_boxes)
    best_gt_for_each_prior = tf.math.reduce_max(ious, axis=0)
    best_gt_for_each_prior_idx = tf.math.argmax(ious, axis=0)
    best_prior_for_each_gt_idx = tf.math.argmax(ious, axis=1)
    best_prior_for_each_gt_idx = np.array(best_prior_for_each_gt_idx)
    best_gt_for_each_prior_idx = np.array(best_gt_for_each_prior_idx)
    best_gt_for_each_prior = np.array(best_gt_for_each_prior)

    for i in range(len(gt_boxes)):
        best_gt_for_each_prior_idx[best_prior_for_each_gt_idx[i]] = i
        best_gt_for_each_prior[best_prior_for_each_gt_idx[i]] = 1

    label_for_each_prior = tf.gather(gt_labels, best_gt_for_each_prior_idx)
    label_for_each_prior = label_for_each_prior.numpy()
    label_for_each_prior[best_gt_for_each_prior < threshold] = 0
    gt_confs = label_for_each_prior
    transformed_boxes = convert_corner_to_center(gt_boxes)
    gt_locs = encode(tf.gather(transformed_boxes, best_gt_for_each_prior_idx), prior_boxes, [0.1, 0.2])
    return tf.convert_to_tensor(gt_confs, dtype=tf.int64), gt_locs

# gt_boxes = tf.constant([[0.244, 0.304, 0.756, 0.73066664]])
# gt_labels = tf.constant([12, 9])
# gt_confs, gt_locs = compute_target(gt_boxes, generate_prior(), gt_labels)
# print(gt_confs)
# print(gt_locs)
