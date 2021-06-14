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
from CentroidTracker import CentroidTracker
import time


class BuildDataset:
    @staticmethod
    def IOU(box1, box2):
        print(box1)
        print(box2)
        # Determine the (x, y) coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute the area of intersection rectangle
        interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = interArea / float(box1Area + box2Area - interArea)

        return iou


    @staticmethod
    def dumpVOCAnnotations(output_folder, filename, size, names, bounding_boxes):
        node_root = Element('annotation')
        
        SubElement(node_root, 'folder').text = 'images'
        SubElement(node_root, 'filename').text = filename + '.jpg'
        SubElement(node_root, 'path').text = os.path.join(output_folder, 'images')
        
        node_size = SubElement(node_root, 'size')
        SubElement(node_size, 'width').text = str(size[1])
        SubElement(node_size, 'height').text = str(size[0])
        SubElement(node_size, 'depth').text = str(size[2])

        SubElement(node_root, 'segmented').text = '0'
        
        for name, bb in zip(names, bounding_boxes):
            node_object = SubElement(node_root, 'object')
            SubElement(node_object, 'name').text = name
            SubElement(node_object, 'pose').text = 'Unspecified'
            SubElement(node_object, 'truncated').text = '0'
            SubElement(node_object, 'difficult').text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            SubElement(node_bndbox, 'xmin').text = str(bb[0])
            SubElement(node_bndbox, 'ymin').text = str(bb[1])
            SubElement(node_bndbox, 'xmax').text = str(bb[2])
            SubElement(node_bndbox, 'ymax').text = str(bb[3])

        tree = ElementTree(node_root)
        tree.write(os.path.join(output_folder, 'annotations', filename) + '.xml', pretty_print=True)


    @staticmethod
    def build(config_path):

        # Obtaining parameters from the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            video_path = config['video_path']
            output_path = config['output_path']
            frame_interval = config['frame_interval']

        # If 'images' and 'annotations' folders already exist, delete them 
        if os.path.exists(os.path.join(output_path, 'images')):
            shutil.rmtree(os.path.join(output_path, 'images'))
        if os.path.exists(os.path.join(output_path, 'annotations')):
            shutil.rmtree(os.path.join(output_path, 'annotations'))
        
        # Creating 'images' and 'annotations' folders
        os.makedirs(os.path.join(output_path, 'images'))
        os.makedirs(os.path.join(output_path, 'annotations'))

        # Creating a CentroidTracker object
        ct = CentroidTracker()

        # Loading YOLOv5s
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # Loading the video
        input_video = cv2.VideoCapture(video_path)

        video_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = int(input_video.get(cv2.CAP_PROP_FPS))
        print('Number of frames in video: ', video_frames)
        print('FPS of video: ', video_fps)


        count = 0
        cent = []
        prev_objects_to_track = {}

        while count <= video_frames - 1:
            ret, frame = input_video.read()
            
            if count % frame_interval == 0:
                detections = model(frame)

                results = detections.pandas().xyxy[0]

                rects = []
                objects = []
                names = []
                for i in range(0, len(results)):
                    if results['confidence'][i] > 0.5:
                        names.append(results['name'][i])
                        bb = [int(results['xmin'][i]), int(results['ymin'][i]), int(results['xmax'][i]), int(results['ymax'][i])]
                        objects.append(bb)
                        #box = np.array([results['xmin'][i], results['ymax'][i], results['xmax'][i], results['ymin'][i]])
                        rects.append(np.array(bb)) #.astype("int")
                
                next_frames_names = names
                next_frames_objects = objects

                dict = {}
                print('objects length:  ', len(objects))
                
                print('prev objects to track:  ')
                print(prev_objects_to_track)
                if len(objects) > 0:
                    curr_objects_to_track = ct.update(rects)
                    print('curr_objects_to_track:  ', curr_objects_to_track)
                    print('objects:  ', objects)
                    for (objectID, centroid), bb in zip(curr_objects_to_track.items(), objects):
                        if objectID in prev_objects_to_track:
                            iou = BuildDataset.IOU(prev_objects_to_track[objectID], bb)
                            print('***IOU:  ', iou)
                            if iou > 0.7:
                                dict[objectID] = prev_objects_to_track[objectID]
                            else:
                                dict[objectID] = bb
                        else:
                            dict[objectID] = bb
                    print('dict:   ', dict)



                    cv2.imwrite(os.path.join(output_path, 'images', str(count)) + '.jpg', frame)
                    BuildDataset.dumpVOCAnnotations(output_path, str(count), frame.shape, names, objects)

                prev_objects_to_track = dict


            else:
                if len(objects) > 0:
                    cv2.imwrite(os.path.join(output_path, 'images', str(count)) + '.jpg', frame)
                    BuildDataset.dumpVOCAnnotations(output_path, str(count), frame.shape, next_frames_names, next_frames_objects)

            count += 1

            print('\n#######################################\n')
            
            # object_id = ct.update(rects)
            # for (objectID, centroid) in object_id.items():
            #         centtemp.append([objectID, centroid[0], centroid[1]])
            # print('centtemp length:  ', len(centtemp))
            # print(centtemp)
            # cent.append(centtemp)
            # centtemp = []
            # # key = cv2.waitKey(1)

        print(count)
        print('Processed all frames')
        print(len(cent))
        print(cent)


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