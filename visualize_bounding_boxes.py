import os
import xml.etree.ElementTree as ET
import cv2 

#### TO BE MODIFIED AS PER NEEDS ####
# Run this script from the folder that contains the 'dir'
dir = 'train'
basenames = ['ck0k9ghqt7a8l0944mcvy0jsx_jpeg.rf.5b55501a403c70b4dba61727046660b5', 
             'ck0kkgwvk9hhg0701edmbvr7r_jpeg.rf.72b0cbcbdd84a2f967e4d56a8a994552', 
             'ck0kmi067lmu70848mrnfi0rc_jpeg.rf.a31f55cd786e5155b79470fb2a7faea0', 
             'ck0lwg0retsfg0848hqi60tlt_jpeg.rf.3025c97e8deb2f2899cbe49461437ae6', 
             'ck0nd98ujhlfg0a46hhco18rf_jpeg.rf.8e239f5ba8045c6acbf54234796128fd']
img_format = '.jpg'
#####################################

cwd = os.getcwd()
xml_dir = os.path.join(cwd, dir)
txt_dir = os.path.join(cwd, 'yolo', dir)

for name in basenames:
    image = cv2.imread(os.path.join(xml_dir, name) + img_format)

    with open(os.path.join(xml_dir, name) + '.xml') as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            if  difficult == 1: continue

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            print('\nCoordinates in VOC format (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

            img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 1)
            cv2.imshow('Image with Bounding Boxes', img)
            cv2.waitKey(0)

    
    with open(os.path.join(txt_dir, name) + '.txt') as txt_file:
        line = txt_file.readline()
        while line:
            class_id, x, y, w, h = map(float, line.split())

            xc = x * img_w
            yc = y * img_h
            xmin = float(xc - 0.5 * w * img_w)
            xmax = float(xc + 0.5 * w * img_w)
            ymin = float(yc - 0.5 * h * img_h)
            ymax = float(yc + 0.5 * h * img_h) 

            print('Coordinates in YOLO format converted to VOC format (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

            img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (0, 0, 255), 1)
            cv2.imshow('Image with Bounding Boxes', img)
            cv2.waitKey(0)

            line = txt_file.readline()