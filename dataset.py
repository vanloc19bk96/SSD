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

    def _get_image(self, index):
        filename = self.ids[index] + '.jpg'
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)

        return image

    def get_annotation(self, index):
        filename = self.ids[index] + '.xml'
        annotation_path = os.path.join(self.annotation_dir, filename)
        objects = ET.parse(annotation_path).findall('object')
        labels = []
        boxes = []
        for object in objects:
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('ymax').text
            ymax = bndbox.find('xmax').text
            labels.append(CLASS_NAME.index(name) + 1)
            boxes.append([xmin, ymin, xmax, ymax])
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
