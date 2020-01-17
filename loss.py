import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

neg_ratio = 3

def hard_negative_mining(loss, gt_confs):
    pos_idx = gt_confs != 0
    num_pos = tf.reduce_sum(tf.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio
    print(loss)
    print(tf.where(pos_idx))
    loss = tf.tensor_scatter_nd_update(loss, tf.where(pos_idx), tf.zeros(pos_idx.shape[0]))
    top = tf.argsort(loss, axis=-1, direction="DESCENDING")
    neg_idx = top < num_neg

    return tf.where(pos_idx), tf.where(neg_idx)

def create_loss(gt_confs, gt_locs, confs, locs):
    cross_entropy = SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    cross_entropy = cross_entropy(gt_confs, confs)
    pos_idx, neg_idx = hard_negative_mining(cross_entropy, gt_confs)
    conf_loss = (tf.reduce_sum(tf.gather_nd(cross_entropy, pos_idx)) + tf.reduce_sum(tf.gather_nd(cross_entropy, neg_idx))) / len(pos_idx)
    smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
    loc_loss = smooth_l1_loss(tf.gather_nd(gt_locs, pos_idx), tf.gather_nd(locs, pos_idx)) / len(pos_idx)
    return conf_loss, loc_loss
