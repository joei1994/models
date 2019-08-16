import cv2
import numpy as np

from object_detection.detector.Detector import Detector

def main():
    frozen_graph_path = '/notebooks/projects/test-object-detection/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb'
    label_map_path = '/notebooks/projects/test-object-detection/object_detection/data/oid_v4_label_map.pbtxt'
    
    image_path = './10.jpg'
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with Detector(frozen_graph_path, label_map_path) as detector:
        output_dict = detector.detect(image)
    print(output_dict)
   
if __name__ == '__main__':
    main()