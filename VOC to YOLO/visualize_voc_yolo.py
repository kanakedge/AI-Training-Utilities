"""
Usage:

(1) From Terminal:
python visualize_voc_yolo.py -c <config_file_path>
OR
python visualize_voc_yolo.py --config_path <config_file_path>

(2) Importing:
from visualize_voc_yolo import VisualizeVOCAndYOLOLabels
VisualizeVOCAndYOLOLabels.visualize(<config_file_path>)
"""

import os
import glob
import xml.etree.ElementTree as ET
import cv2 
import yaml
import sys
import getopt

class VisualizeVOCAndYOLOLabels:
    img_dir_path = None
    voc_dir_path = None
    yolo_dir_path = None
    img_format = None

    @staticmethod
    def visualizeBB(basenames):
        for name in basenames:
            image = cv2.imread(os.path.join(VisualizeVOCAndYOLOLabels.img_dir_path, name) + VisualizeVOCAndYOLOLabels.img_format)

            with open(os.path.join(VisualizeVOCAndYOLOLabels.voc_dir_path, name) + '.xml') as xml_file:
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

                    print('\nVOC Coordinates (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

                    img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 1)
                    cv2.imshow('Image with Bounding Boxes', img)
                    cv2.waitKey(0)

            
            with open(os.path.join(VisualizeVOCAndYOLOLabels.yolo_dir_path, name) + '.txt') as txt_file:
                line = txt_file.readline()
                while line:
                    class_id, x, y, w, h = map(float, line.split())

                    xc = x * img_w
                    yc = y * img_h
                    xmin = float(xc - 0.5 * w * img_w)
                    xmax = float(xc + 0.5 * w * img_w)
                    ymin = float(yc - 0.5 * h * img_h)
                    ymax = float(yc + 0.5 * h * img_h) 

                    print('YOLO Coordinates (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

                    img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (0, 0, 255), 1)
                    cv2.imshow('Image with Bounding Boxes', img)
                    cv2.waitKey(0)

                    line = txt_file.readline()


    @staticmethod
    def visualize(config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            VisualizeVOCAndYOLOLabels.img_dir_path = config['img_dir_path']
            VisualizeVOCAndYOLOLabels.voc_dir_path = config['voc_dir_path']
            VisualizeVOCAndYOLOLabels.yolo_dir_path = config['yolo_dir_path']
            basenames = config['basenames']
            VisualizeVOCAndYOLOLabels.img_format = config['img_format']


        if not basenames:
            basename_list = []
            for filename in glob.glob(os.path.join(VisualizeVOCAndYOLOLabels.img_dir_path, '*') + VisualizeVOCAndYOLOLabels.img_format):
                basename = os.path.basename(filename)
                basename_no_ext = os.path.splitext(basename)[0]
                basename_list.append(basename_no_ext)
            
            VisualizeVOCAndYOLOLabels.visualizeBB(basename_list)
            
        else:
            VisualizeVOCAndYOLOLabels.visualizeBB(basenames)


# If the script is run from the terminal
if __name__ == '__main__':
    try:
        arguments, values = getopt.getopt(sys.argv[1:], 'c:', ['config_path='])

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-c", "--config_path"):
                config_path = currentValue

        VisualizeVOCAndYOLOLabels.visualize(config_path)

    except getopt.GetoptError as err:
        print(str(err))     