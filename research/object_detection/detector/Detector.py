import tensorflow as tf
from object_detection.utils import ops as utils_ops
import cv2
import numpy as np
import os
import six.moves.urllib as urllib
from PIL import Image
import sys
import tarfile
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util

from pdb import set_trace

class Detector:
    def __init__(self, frozen_graph_path, label_map_path):
        if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
            
        self.frozen_graph_path = frozen_graph_path
        self.label_map_path = label_map_path
        
        # init graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            graphDef = tf.GraphDef()
            with tf.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serializedGraph = fid.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name='')
    
        # Loading category index
        self.categoryIndex = label_map_util.create_category_index_from_labelmap(self.label_map_path, use_display_name=True)
        
        self.session = None
        
    def __enter__(self):   
        self.openSession()
        return self
    
    def __exit__(self, *args):
        self.closeSession()
        
    def _runInferenceForSingleImage(self, image):
        image = np.expand_dims(image, axis=0)
        
        # Get handles to input and output tensors
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(
                tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(
                tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(
                tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                    real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                    real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[1], image.shape[2])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = self.session.run(tensor_dict,feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(
            output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    
    def _chooseBestDetection(self,
                           image,
                           boxes, 
                           classes, 
                           scores,
                           categoryIndex,
                           maxDediction,                           
                           minScoreThresh):
        detections = []
        height, width, _ = image.shape
        for i in range(min(boxes.shape[0], maxDediction)):
            if scores[i] > minScoreThresh:
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                xmin, ymin, xmax, ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

                if classes[i] in categoryIndex.keys():
                    label = categoryIndex[classes[i]]['name']
                else:
                    label = 'NA'
                score = scores[i]

                detection = {}
                detection['box'] = (xmin, ymin, xmax, ymax)
                detection['label'] = label
                detection['score'] = score
                detections.append(detection)
        return detections
    
    def openSession(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config) if (self.session is None) else self.session
    
    def closeSession(self):
        if self.session is not None:
            self.session.close()
    
    def detect(self, image, maxDetection = 1, minScoreThreshold = .5):
        outputDict = self._runInferenceForSingleImage(image)
        detections = self._chooseBestDetection(
            image,
            outputDict['detection_boxes'], 
            outputDict['detection_classes'], 
            outputDict['detection_scores'],
            self.categoryIndex,
            maxDetection,
            minScoreThreshold
        )
        return detections
