import numpy as np
import cv2
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


class ClothesModelService(object):
    def __init__(self, input_size=416, weight='./checkpoints/clothes-yolov4-tiny-416', iou=0.5, score=0.5):
        self.input_size = input_size
        self.weight = weight
        self.savedModelLoaded = tf.saved_model.load(
            weight, tags=[tag_constants.SERVING])
        self.infer = self.savedModelLoaded.signatures['serving_default']
        self.iou = iou
        self.score = score
        pass

    def detectClothes(self, frame):
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        pred_bbox = [bboxes, scores, classes, num_objects]
        print("Number of Object: ", num_objects)
        print("Classes: ", classes)
        return classes
