import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

CLASS_NAME = [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor']

class Dataset(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')
        self.ids = [name[:-4] for name in os.listdir(self.image_dir)]
        self.train_ids = self.ids[:int(0.75*len(self.ids))]
        self.valid_ids = self.ids[int(len(self.ids)*0.75):]

    def _get_image(self, ids, index):
        filename = ids[index] + '.jpg'
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)

        return image

    def get_annotation(self, ids, index, image_shape):
        filename = ids[index] + '.xml'
        annotation_path = os.path.join(self.annotation_dir, filename)
        objects = ET.parse(annotation_path).findall('object')
        labels = []
        boxes = []
        w, h = image_shape
        for object in objects:
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = float(bndbox.find('xmin').text) / w
            ymin = float(bndbox.find('ymin').text) / h
            xmax = float(bndbox.find('ymax').text) / w
            ymax = float(bndbox.find('xmax').text) / h
            labels.append(CLASS_NAME.index(name) + 1)
            boxes.append([xmin, ymin, xmax, ymax])
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
