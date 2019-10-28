import glob
import os
import xml.etree.ElementTree as ET

'''
    This function creates label_map.pbtxt from PascalVOC xml
     and find max of images width and height

    parameter 1 : Path to directory of PascalVOC xml
    parameter 2 : Path to label_map.pbtxt file to create
'''

def create_label_map(xml_dir, output_label_map_path, target_classes):
    max_image_width = 0
    max_image_height = 0
    
    xml_paths = glob.glob(xml_dir + '/*.xml', recursive=True)
    
    labels = set()
    for xml_path in xml_paths:
        with open(xml_path, 'r', encoding='UTF-8') as fid:
            tree = ET.fromstring(fid.read())
        # get width/height of image
        size = [child for child in tree.iter('size')][0]
        w, h = int(size[0].text), int(size[1].text)
        # find max of image width/height
        if w > max_image_width: max_image_width = w
        if h > max_image_height: max_image_height = h    

        objs = [child for child in tree.iter('object')]
        for obj in objs:
            name = [name for name in obj.iter('name')]
            label = name[0].text
            #if ord(label) in range(ord('ก'), ord('ฮ') + 1):
            #    labels.add(ord(label))
            #else:
            #    labels.add(label)
            if (target_classes != None) and (label not in target_classes):
                continue
            labels.add(name[0].text)
    
    # create label_map and save
    for i, label in enumerate(labels):
        out = ''
        out += 'item' + ' ' + '{' + '\n'
        out += '  ' + 'id: ' + (str(i+1)) + '\n'
        out += '  ' + 'name: ' + '\'' + str(label) + '\'' + '\n'
        out += '}' + '\n\n'

        with open(output_label_map_path, 'ab') as f:
            f.write(out.encode('utf-8'))
    
    print(f"max_width: {max_image_width}, max_height: {max_image_height}")
    
    
    