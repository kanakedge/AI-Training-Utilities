"""
Usage:

(1) From Terminal:
python build_dataset.py -c <config_file_path>
OR
python build_dataset.py --config_path <config_file_path>

(2) Importing:
from build_dataset import BuildDataset
BuildDataset.build(<config_file_path>)
"""

import os
import sys
import getopt
import shutil
import yaml
import cv2
import torch
import numpy as np
from lxml.etree import Element, SubElement, ElementTree
#from CentroidTracker import CentroidTracker


class BuildDataset:
    @staticmethod
    def dumpVOCAnnotations(output_folder, filename, size, objects):
        node_root = Element('annotation')
        
        SubElement(node_root, 'folder').text = 'images'
        SubElement(node_root, 'filename').text = filename + '.jpg'
        SubElement(node_root, 'path').text = os.path.join(output_folder, 'images')
        
        node_size = SubElement(node_root, 'size')
        SubElement(node_size, 'width').text = str(size[1])
        SubElement(node_size, 'height').text = str(size[0])
        SubElement(node_size, 'depth').text = str(size[2])

        SubElement(node_root, 'segmented').text = '0'
        
        for object in objects:
            node_object = SubElement(node_root, 'object')
            SubElement(node_object, 'name').text = object[0]
            SubElement(node_object, 'pose').text = 'Unspecified'
            SubElement(node_object, 'truncated').text = '0'
            SubElement(node_object, 'difficult').text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            SubElement(node_bndbox, 'xmin').text = str(object[1])
            SubElement(node_bndbox, 'ymin').text = str(object[2])
            SubElement(node_bndbox, 'xmax').text = str(object[3])
            SubElement(node_bndbox, 'ymax').text = str(object[4])

        tree = ElementTree(node_root)
        tree.write(os.path.join(output_folder, 'annotations', filename) + '.xml', pretty_print=True)


    @staticmethod
    def build(config_path):
        # Obtaining parameters from the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            video_path = config['video_path']
            output_path = config['output_path']

        # If 'images' and 'annotations' folders already exist, delete them 
        if os.path.exists(os.path.join(output_path, 'images')):
            shutil.rmtree(os.path.join(output_path, 'images'))
        if os.path.exists(os.path.join(output_path, 'annotations')):
            shutil.rmtree(os.path.join(output_path, 'annotations'))
        
        # Creating 'images' and 'annotations' folders
        os.makedirs(os.path.join(output_path, 'images'))
        os.makedirs(os.path.join(output_path, 'annotations'))

        #ct = CentroidTracker()

        # Loading YOLOv5s
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Loading the video
        input_video = cv2.VideoCapture(video_path)

        no_of_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 1
        # cent = []
        while count <= no_of_frames:
            ret, frame = input_video.read()

            detections = model(frame)
            results = detections.pandas().xyxy[0]

            #rects = []
            objects = []
            for i in range(0, len(results)):
                    if results['confidence'][i] > 0.5:
                        objects.append([results['name'][i], int(results['xmin'][i]), int(results['ymin'][i]), int(results['xmax'][i]), int(results['ymax'][i])])
                        #box = np.array([results['xmin'][i], results['ymax'][i], results['xmax'][i], results['ymin'][i]])
                        #rects.append(box.astype("int"))

            if len(objects) > 0:
                cv2.imwrite(os.path.join(output_path, 'images', str(count)) + '.jpg', frame)
                BuildDataset.dumpVOCAnnotations(output_path, str(count), frame.shape, objects)

            count += 1
            
            # objects = ct.update(rects)
            # for (objectID, centroid) in objects.items():
            #         cent.append([objectID, centroid[0], centroid[1]])
            # key = cv2.waitKey(1)

        print('Processed all frames')



# If the script is run from the terminal
if __name__ == '__main__':
    try:
        arguments, values = getopt.getopt(sys.argv[1:], 'c:', ['config_path='])

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-c", "--config_path"):
                config_path = currentValue

        BuildDataset.build(config_path)

    except getopt.GetoptError as err:
        print(str(err)) 