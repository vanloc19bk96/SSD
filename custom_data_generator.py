import cv2
from dataset import Dataset
from compute_target import *
from augmentation import augment
from prior_generation import *
import tensorflow as tf
from functools import partial

def generate(root_dir, prior_boxes, new_size=300, subset=None, shuffle=False, augmentation=False):
    voc = Dataset(root_dir)
    if subset == 'train':
        indices = voc.train_ids
    elif subset == 'valid':
        indices = voc.valid_ids
    else:
        indices = voc.ids

    if shuffle:
        indices = np.random.shuffle()
    for index in range(len(indices)):
        image = voc._get_image(indices, index)
        w, h, _ = image.shape
        image = cv2.resize(image, (new_size, new_size))
        image = image / 255
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        gt_boxes, gt_labels = voc.get_annotation(indices, index, (w, h))
        if augmentation:
            image, gt_boxes, gt_labels = augment(image, gt_boxes, gt_labels)
        gt_confs, gt_locs = compute_target(gt_boxes, prior_boxes, gt_labels)
        yield image, gt_confs, gt_locs

def create_batch_generator(root_dir, prior_boxes, batch_size, new_size=300, subset=None, shuffle=False, augmentation=False):
    if subset == 'train':
        train_generator = partial(generate, root_dir, prior_boxes, new_size, 'train', shuffle, augmentation)
        train_dataset = tf.data.Dataset.from_generator(train_generator, (tf.float32, tf.int32, tf.float32)).batch(batch_size)
        valid_generator = partial(generate, root_dir, prior_boxes, new_size, 'valid', False, False)
        valid_dataset = tf.data.Dataset.from_generator(valid_generator, (tf.float32, tf.int32, tf.float32)).batch(batch_size)
        return train_dataset.take(-1), valid_dataset.take(-1)
    else:
        generator = partial(generate, root_dir, prior_boxes, new_size, shuffle, augmentation)
        dataset = tf.data.Dataset.from_generator(
            generator, (tf.string, tf.float32, tf.int64, tf.float32))
        dataset = dataset.batch(batch_size)
        return dataset.take(-1)

# priors = generate_prior()
# dt = create_batch_generator(r"C:\Users\Admin\Desktop\data", priors, 8, shuffle=True, augmentation=True)