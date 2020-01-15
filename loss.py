import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

neg_ratio = 3

def hard_negative_mining(loss, gt_confs):
    pos_idx = tf.where(gt_confs != 0)
    num_neg = len(pos_idx) * neg_ratio
    neg_loss_confs = tf.identity(loss)
    neg_loss_confs = tf.tensor_scatter_nd_update(neg_loss_confs, pos_idx,
                                                 tf.zeros(pos_idx.shape[0]))

    tmp = tf.reshape(neg_loss_confs, [1, -1])
    top = tf.argsort(tmp, axis=-1, direction="DESCENDING")
    top = top < num_neg
    batch, h, w, num_boxes = gt_confs.shape
    top = tf.reshape(top, [batch, h, w, num_boxes])
    neg_idx = tf.where(top)

    return pos_idx, neg_idx


def loss(gt_confs, gt_locs, prediction):
    locs = prediction[:, :, :, :, :4]
    confs = prediction[:, :, :, :, 4:]
    batch, h, w, num_boxes = confs.shape[:-1]
    gt_confs = tf.reshape(gt_confs, [batch, h, w, num_boxes])
    gt_locs = tf.reshape(gt_locs, [batch, h, w, num_boxes, -1])

    cross_entropy = SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    cross_entropy = cross_entropy(gt_confs, confs)
    pos_idx, neg_idx = hard_negative_mining(cross_entropy, gt_confs)
    conf_loss = (tf.reduce_sum(tf.gather_nd(cross_entropy, pos_idx)) + tf.reduce_sum(tf.gather_nd(cross_entropy, neg_idx))) / len(pos_idx)
    smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')
    loc_loss = smooth_l1_loss(tf.gather_nd(gt_locs, pos_idx), tf.gather_nd(locs, pos_idx)) / len(pos_idx)

    return conf_loss, loc_loss

# prediction = tf.random.uniform(shape=(2, 3, 3, 4, 7), minval=0, maxval=10, dtype=tf.dtypes.float32)
# locs = tf.random.uniform(shape=(72, 4))
# confs = tf.zeros(shape=(72,))
# confs = confs.numpy()
# confs[5] = 0.9
# confs[4] = 0.2
# print(loss(tf.convert_to_tensor(confs), locs, prediction)[1])
