"""
Usage:

(1) From Terminal:
python visualize_coco_yolo.py -c <config_file_path>
OR
python visualize_coco_yolo.py --config_path <config_file_path>

(2) Importing:
from visualize_coco_yolo import VisualizeCOCOAndYOLOLabels
VisualizeCOCOAndYOLOLabels.visualize(<config_file_path>)
"""

import os
import glob
import json
import cv2 
import yaml
import sys
import getopt

class VisualizeCOCOAndYOLOLabels:
    img_dir_path = None
    coco_dir_path = None
    yolo_dir_path = None
    img_format = None

    @staticmethod
    def visualizeBB(basenames):
        # Reading data from JSON file
        json_file = glob.glob(os.path.join(VisualizeCOCOAndYOLOLabels.coco_dir_path, '*') + '.json')[0]
        with open(json_file) as f:
            data = json.load(f)

        for name in basenames:
            image = cv2.imread(os.path.join(VisualizeCOCOAndYOLOLabels.img_dir_path, name) + VisualizeCOCOAndYOLOLabels.img_format)

            for img in data['images']:
                if img['file_name'] == name + VisualizeCOCOAndYOLOLabels.img_format:
                    img_id = img['id']
                    img_w = img['width']
                    img_h = img['height']
                    break

            for ann in data['annotations']:
                if ann['id'] == img_id:
                    xmin = ann['bbox'][0]
                    ymin = ann['bbox'][1]
                    w = ann['bbox'][2]
                    h = ann['bbox'][3]
                    break
            
            xmax = xmin + w
            ymax = ymin + h
            print('\nCOCO Coordinates (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

            img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 1)
            cv2.imshow('Image with Bounding Boxes', img)
            cv2.waitKey(0)

            
            with open(os.path.join(VisualizeCOCOAndYOLOLabels.yolo_dir_path, name) + '.txt') as txt_file:
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
            VisualizeCOCOAndYOLOLabels.img_dir_path = config['img_dir_path']
            VisualizeCOCOAndYOLOLabels.coco_dir_path = config['coco_dir_path']
            VisualizeCOCOAndYOLOLabels.yolo_dir_path = config['yolo_dir_path']
            basenames = config['basenames']
            VisualizeCOCOAndYOLOLabels.img_format = config['img_format']


        if not basenames:
            basename_list = []
            for filename in glob.glob(os.path.join(VisualizeCOCOAndYOLOLabels.img_dir_path, '*') + VisualizeCOCOAndYOLOLabels.img_format):
                basename = os.path.basename(filename)
                basename_no_ext = os.path.splitext(basename)[0]
                basename_list.append(basename_no_ext)
            
            VisualizeCOCOAndYOLOLabels.visualizeBB(basename_list)
            
        else:
            VisualizeCOCOAndYOLOLabels.visualizeBB(basenames)


# If the script is run from the terminal
if __name__ == '__main__':
    try:
        arguments, values = getopt.getopt(sys.argv[1:], 'c:', ['config_path='])

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-c", "--config_path"):
                config_path = currentValue

        VisualizeCOCOAndYOLOLabels.visualize(config_path)

    except getopt.GetoptError as err:
        print(str(err))    