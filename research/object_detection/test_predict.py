import cv2
import numpy as np
from glob import glob
import os
import time
from pdb import set_trace

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
    
def detectImages(detector, imagePaths, output_dir):
    for image_path in imagePaths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect(image)
        imageWithBbox = drawBbox(image, detections)
        imageName = image_path.split('/')[-1].split('.')[0]
        saveImage(imageName, imageWithBbox, output_dir)

def detectVideo(detector, videoPath):
    cap = cv2.VideoCapture(videoPath)
    print(f"fps : {cap.get(5)}")
    print(f"total frames : {cap.get(7)}")
    
    while(True):
        ret, frame = cap.read()
        frameIndex = int(cap.get(0))
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        startTime = time.time()
        detections = detector.detect(frame)
        endTime = time.time()
        diffTime = endTime - startTime
        print(f"fps: {1/(diffTime)}")
        #imageWithBbox = drawBbox(frame, detections)
        #saveImage(str(frameIndex), imageWithBbox, './test_images/result')
        cv2.waitKey(27)
    cap.release()
    
def main():
    frozen_graph_path = './ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
    label_map_path = './data/mscoco_label_map.pbtxt'
    
    image_dir = './test_images/'
    image_paths = [image for image in glob(image_dir + '*.jpg')]
    output_dir = './test_images/result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    videoPath = './test_images/detect-plate/cars.mp4'

    with Detector(frozen_graph_path, label_map_path) as detector:
        detectImages(detector, videoPath, output_dir)

   
if __name__ == '__main__':
    main()