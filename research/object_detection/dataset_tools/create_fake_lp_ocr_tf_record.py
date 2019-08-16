import hashlib
import io
import logging
import os 
import random
import glob
from pdb import set_trace

import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('images_dir', '/notebooks/projects/object-detection/workspace/fake-license-plate-ocr/data/train_set/better_gen_trainset', 'Path to images directory')
flags.DEFINE_string('xmls_dir', '/notebooks/projects/object-detection/workspace/fake-license-plate-ocr/data/train_set/better_gen_trainset', 'Path to xmls directory')
train_ratio = .98
model_number = 4
flags.DEFINE_string('label_map_path', 
                    f'/notebooks/projects/object-detection/workspace/fake-license-plate-ocr/training/model_{model_number}/label_map/label_map.pbtxt', 'Path to label_map file')
flags.DEFINE_string('output_dir',
                    f'/notebooks/projects/object-detection/workspace/fake-license-plate-ocr/training/model_{model_number}/data', 'Path to output file')

flags.DEFINE_integer('num_shards', 10, 'Number of shards')

FLAGS = flags.FLAGS

def is_bbox_valid(xmin, ymin, xmax, ymax, w, h, image_name):
    #ignore if bounding box size is too small
    obj_area = (xmax - xmin) * (ymax - ymin)
    if obj_area < ((w * h) / (16*16)):
        logging.warning(f'Too small : {image_name}.jpg {xmin}, {ymin}, {xmax}, {ymax}')
        return False
    #verify bounding box out side image
    if (xmin < 0 or xmin > 600 or xmax < 0 or xmax > 600 or ymin < 0 or ymin > 200 or ymax < 0 or ymax > 200):
        logging.warning(f'Out of image border : {image_name}.jpg {xmin}, {ymin}, {xmax}, {ymax}')
        return False
    
    return True

def get_bbox_vertices(bbox, w, h):
    xmin = float(bbox['xmin'])
    ymin = float(bbox['ymin'])
    xmax = float(bbox['xmax'])
    ymax = float(bbox['ymax'])
    
    if xmin < 0: 
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > w:
        xmax = w
    if ymax > h:
        ymax = h
        
    return xmin, ymin, xmax, ymax    

def get_file_name(image_path):
    return image_path.split('/')[-1].split('.')[0]

def create_tf_example(image_path,
                      xml_path,
                      label_map_dict):
    #read image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
        
    #create hash    
    key = hashlib.sha256(encoded_jpg).hexdigest()
    
    #read xml annotation
    with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    xml_dict = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    
    #width = int(xml_dict['size']['width'])
    #height = int(xml_dict['size']['height'])
    width, height = image.size
    
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    classes = []
    classes_text = []
    
    if 'object' in xml_dict:

        for obj in xml_dict['object']:
            xmin, ymin, xmax, ymax = get_bbox_vertices(obj['bndbox'], width, height)    
            
            #verfiy bbox
            if not is_bbox_valid(xmin, ymin, xmax, ymax,
                                 width,
                                 height, 
                                 get_file_name(image_path)):
                continue
                
                
            xmins.append(xmin / width)
            ymins.append(ymin / height)
            xmaxs.append(xmax / width)
            ymaxs.append(ymax / height)
    
            class_name = obj['name']
            #if ord(class_name) in range(ord('ก'), ord('ฮ') + 1):
            #    class_name = str(ord(class_name))
            
            classes_text.append(class_name.encode('utf8'))
            classes.append(label_map_dict[class_name])

            
    feature_dict = {
        'image/height' : dataset_util.int64_feature(height),
        'image/width' : dataset_util.int64_feature(width),
        'image/filename' : dataset_util.bytes_feature(xml_dict['filename'].encode('utf8')),
        'image/source_id' : dataset_util.bytes_feature(xml_dict['filename'].encode('utf8')),
        'image/key/sha256' : dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded' : dataset_util.bytes_feature(encoded_jpg),
        'image/format' : dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin' : dataset_util.float_list_feature(xmins),
        'image/object/bbox/ymin' : dataset_util.float_list_feature(ymins),
        'image/object/bbox/xmax' : dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymax' : dataset_util.float_list_feature(ymaxs),
        'image/object/bbox/class/text' : dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def create_tf_record(xmls_dir,
                    images_dir,
                    image_names,
                    label_map_dict,
                    output_filename,
                    num_shards): 
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, output_filename, num_shards)
        for idx, image_name in enumerate(image_names):
            #define xml_path
            xml_path = os.path.join(xmls_dir, image_name + '.xml')
            
            if not os.path.exists(xml_path):
                logging.info(f'Cound not find {xml_path}')
                continue
                
            #define image_path    
            image_path = os.path.join(images_dir, image_name + '.jpg')
            if not os.path.exists(image_path):
                logging.info(f'Cound not find {image_path}')
                continue
                
            try:
                tf_example = create_tf_example(image_path, xml_path, label_map_dict)
                if tf_example:
                    shard_idx = idx % num_shards
                    output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            except ValueError:
                logging.warning('Invalid example: %s, ignoring.', xml_path)
                
def main(_):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    image_names = [get_file_name(f) for f in glob.glob(FLAGS.images_dir + '**/*.jpg')]
    
    random.seed(42)
    random.shuffle(image_names)
    num_images = len(image_names)
    num_train = int(train_ratio * num_images)
    
    #train-test split
    train_images = image_names[:num_train]
    val_images = image_names[num_train:]
    
    #create output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    
    
    with open(os.path.join(FLAGS.output_dir + 'source.txt'), 'a') as fid:
        fid.write(f'images_dir : {FLAGS.images_dir}')
        fid.write(f'\n xmls_dir : {FLAGS.xmls_dir}')
        fid.write(f'\n num_train : {num_images}\n train_ratio : {train_ratio}')
    
    #define output path
    train_output_path = os.path.join(FLAGS.output_dir, 'fake_lp_ocr_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fake_lp_ocr_val.record')
    
    create_tf_record(
        FLAGS.xmls_dir,
        FLAGS.images_dir,
        train_images,
        label_map_dict,
        train_output_path,
        FLAGS.num_shards
    )
    
    create_tf_record(
        FLAGS.xmls_dir,
        FLAGS.images_dir,
        val_images,
        label_map_dict,
        val_output_path,
        FLAGS.num_shards
    )
    
if __name__ == '__main__':
    tf.app.run()