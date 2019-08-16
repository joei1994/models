import cv2
import numpy as np
from glob import glob
import os

from detector.Detector import Detector

def drawBbox(image, detections):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for det in detections:
        xmin, ymin, xmax, ymax = det['box']
        label = det['label']
        cv2.rectangle(image, 
                      (xmin, ymin), 
                      (xmax, ymax), 
                      (255, 0, 0))
    return image

def saveImage(imageName, image, outputDir):
    outputPath = os.path.join(outputDir, imageName + '.jpg')
    cv2.imwrite(outputPath, image)
    

def main():
    frozen_graph_path = './faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12/frozen_inference_graph.pb'
    label_map_path = './data/oid_v4_label_map.pbtxt'
    
    image_dir = './test_images/detect-plate'
  
    image_paths = [image for image in glob(image_dir + '**/*.jpg')]
    
    with Detector(frozen_graph_path, label_map_path) as detector:
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect(image)
            
            imageWithBbox = drawBbox(image, detections)
            imageName = image_path.split('/')[-1].split('.')[0]
            saveImage(imageName, imageWithBbox, './test_images/result')
   
if __name__ == '__main__':
    main()