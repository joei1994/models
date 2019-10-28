import hashlib
import io
import logging
import os 
import random
import glob
import argparse
from pdb import set_trace

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def is_bbox_valid(xmin, ymin, xmax, ymax, image_width, image_height, image_name):
    # ignore if bounding box size is too small
    """
    obj_area = (xmax - xmin) * (ymax - ymin)
    if obj_area < ((image_width * image_height) / (16*16)):
        logging.warning(f'Too small : {image_name}.jpg {xmin}, {ymin}, {xmax}, {ymax}')
        return False
    """    
    # verify bounding box out side image
    if (xmin < 0 or xmin > image_width or
        xmax < 0 or xmax > image_width or
        ymin < 0 or ymin > image_height or 
        ymax < 0 or ymax > image_height):
        
        logging.warning(f'Out of image border : {image_name}.jpg {xmin}, {ymin}, {xmax}, {ymax}')
        return False
    
    return True

def get_bbox_vertices(bbox):
    xmin = float(bbox['xmin'])
    ymin = float(bbox['ymin'])
    xmax = float(bbox['xmax'])
    ymax = float(bbox['ymax'])

    return xmin, ymin, xmax, ymax    

def get_file_name(image_path):
    return image_path.split('/')[-1].split('.')[0]

def create_tf_example(image_path,
                      xml_path,
                      label_map_dict,
                      target_classes):
    # read image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    # create hash    
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
        
    # read xml annotation
    with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    xml_dict = dataset_util.recursive_parse_xml_to_dict(xml)
    
    width, height = image.size
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    annotationElem = xml_dict['annotation']
    if 'object' in annotationElem:
        for obj in annotationElem['object']:
            class_name = obj['name']
            if (target_classes != None) and (class_name.lower() not in [tc.lower() for tc in target_classes]):
                continue
            xmin, ymin, xmax, ymax = get_bbox_vertices(obj['bndbox'])    
            
            #verfiy bbox
            #if not is_bbox_valid(xmin, ymin, xmax, ymax, width, height, get_file_name(image_path)): continue
                            
            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
    
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])

    feature_dict = {
        'image/height' : dataset_util.int64_feature(height),
        'image/width' : dataset_util.int64_feature(width),
        'image/filename' : dataset_util.bytes_feature(annotationElem['filename'].encode('utf8')),
        'image/source_id' : dataset_util.bytes_feature(annotationElem['filename'].encode('utf8')),
        'image/key/sha256' : dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
        'image/format' : dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
        'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
        'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
        'image/object/class/text' : dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def create_tf_record(xmls_dir,
                    images_dir,
                    image_names,
                    label_map_dict,
                    output_filename,
                    num_shards,
                    target_classes): 
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_filename, num_shards)
        for idx, image_name in enumerate(image_names):
            #define xml_path
            xml_file = os.path.join(xmls_dir, image_name + '.xml')
            
            if not os.path.exists(xml_file):
                logging.info(f'Cound not find {xml_file}')
                continue
                
            #define image_path    
            image_file = os.path.join(images_dir, image_name + '.jpg')
            if not os.path.exists(image_file):
                logging.info(f'Cound not find {image_file}')
                continue
                
            try:
                tf_example = create_tf_example(image_file, xml_file, label_map_dict, target_classes)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', xml_file)
                
def generate_tf_records(dataset_dir, label_map_file, tf_record_dir, train_ratio=.8, num_shards=10, target_classes=None):
    label_map_dict = label_map_util.get_label_map_dict(label_map_file)
    image_names = [get_file_name(f) for f in glob.glob(dataset_dir + '**/*.jpg')]
    

    num_images = len(image_names)
    num_train = int(train_ratio * num_images)

    random.seed(42)
    random.shuffle(image_names)
    
    #train-test split
    train_images = image_names[:num_train]
    val_images = image_names[num_train:]
    
    #create output dir
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)
    
    with open(os.path.join(tf_record_dir + 'source.txt'), 'a') as fid:
        fid.write(f'tf_record_dir : {tf_record_dir}')
        fid.write(f'\nnum_train : {num_images}\n train_ratio : {train_ratio}')
    
    #define output path
    train_output_path = os.path.join(tf_record_dir, 'train.record')
    val_output_path = os.path.join(tf_record_dir, 'validate.record')
    
    create_tf_record(
        dataset_dir,
        dataset_dir,
        train_images,
        label_map_dict,
        train_output_path,
        num_shards,
        target_classes
    )
    
    create_tf_record(
        dataset_dir,
        dataset_dir,
        val_images,
        label_map_dict,
        val_output_path,
        num_shards,
        target_classes
    )
    
if __name__ == '__main__':
    tf.app.run()