import os
import tensorflow as tf
from ssd import create_ssd
from prior_generation import generate_prior
from custom_data_generator import create_batch_generator
from loss import create_loss


checkpoint = 'checkpoint'
root_dir = r'C:\Users\Admin\Desktop\data'
batch_size = 1
NUM_CLASSES = 21
epochs = 20
def grad(images, gt_locs, gt_confs, ssd, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(images)
        conf_loss, loc_loss = create_loss(gt_locs, gt_confs, confs, locs)
        loss = conf_loss + loc_loss
    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss

if __name__ == "__main__":
    os.makedirs(checkpoint, exist_ok=True)

    prior_boxes = generate_prior()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    ssd = create_ssd(NUM_CLASSES)
    batch_generator, valid_generator = create_batch_generator(root_dir, prior_boxes, batch_size, subset='train', augmentation=True)
    for epoch in range(epochs):
        total_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        for i, (images, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss = grad(images, gt_confs, gt_locs, ssd, optimizer)
            total_loss += loss.numpy()
        print("Epoch {}: {}".format(epoch, total_loss/17125))