import glob
import os
import xml.etree.ElementTree as ET

def create_label_map(xml_dir, label_map_path):
    max_width = 0
    max_height = 0
    
    xml_paths = glob.glob(xml_dir + '/**/*.xml', recursive=True)
    
    labels = set()
    for xml_path in xml_paths:
        with open(xml_path, 'r', encoding='UTF-8') as fid:
            tree = ET.fromstring(fid.read())
        size = [child for child in tree.iter('size')][0]
        w, h = int(size[0].text), int(size[1].text)

        if w > max_width: max_width = w
        if h > max_height: max_height = h    

        objs = [child for child in tree.iter('object')]
        for obj in objs:
            name = [name for name in obj.iter('name')]
            label = name[0].text
            #if ord(label) in range(ord('ก'), ord('ฮ') + 1):
            #    labels.add(ord(label))
            #else:
            #    labels.add(label)
            labels.add(name[0].text)
            
    for i, label in enumerate(labels):
        out = ''
        out += 'item' + ' ' + '{' + '\n'
        out += '  ' + 'id: ' + (str(i+1)) + '\n'
        out += '  ' + 'name: ' + '\'' + str(label) + '\'' + '\n'
        out += '}' + '\n\n'

        with open(label_map_path, 'ab') as f:
            f.write(out.encode('utf-8'))
    
    print(f"max_width: {max_width}, max_height: {max_height}")
    
    
    