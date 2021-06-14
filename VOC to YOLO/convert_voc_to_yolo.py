"""
Usage:

(1) From Terminal:
python convert_voc_to_yolo.py -c <config_file_path>
OR
python convert_voc_to_yolo.py --config_path <config_file_path>

(2) Importing:
from convert_voc_to_yolo import ConvertVOCToYOLO
ConvertVOCToYOLO.convert(<config_file_path>)
"""

import os
import glob
import xml.etree.ElementTree as ET
import yaml
import sys
import getopt
import random
import numpy as np
import shutil

class ConvertVOCToYOLO:
    classes = []

    @staticmethod
    def convertBB(img_w, img_h, xmin, xmax, ymin, ymax):
        # Converting VOC format to YOLO format
        x = (xmin + xmax)/2/img_w
        y = (ymin + ymax)/2/img_h
        w = (xmax - xmin)/img_w
        h = (ymax - ymin)/img_h

        return (x, y, w, h)

    @staticmethod
    def convert_annotation(basename_no_ext, dir_path, output_path):
        with open(os.path.join(dir_path, basename_no_ext) + '.xml') as in_file, open(os.path.join(output_path, basename_no_ext) + '.txt', 'w') as out_file:
            tree = ET.parse(in_file)
            root = tree.getroot()

            # Obtaining width and height of the image from the XML file
            size = root.find('size')
            img_w = int(size.find('width').text)
            img_h = int(size.find('height').text)

            # Iterating through all the bounding boxes in the image
            for obj in root.iter('object'):
                difficult = int(obj.find('difficult').text)
                if difficult == 1:
                    continue

                # If class in XML file not already in class names list, add it
                class_name = obj.find('name').text
                if class_name not in ConvertVOCToYOLO.classes:
                    ConvertVOCToYOLO.classes.append(class_name)
                

                # From the class name given in the XML file, obtaining the class ID
                class_id = ConvertVOCToYOLO.classes.index(class_name)

                # Obtaining xmin, xmax, ymin, ymax of the bounding box
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymin = float(xmlbox.find('ymin').text)
                ymax = float(xmlbox.find('ymax').text)

                # Converting to YOLO format
                annotation = ConvertVOCToYOLO.convertBB(img_w, img_h, xmin, xmax, ymin, ymax)

                # Writing the YOLO format bounding box in a .txt file
                out_file.write(str(class_id) + " " + " ".join([str(val) for val in annotation]) + '\n')


    @staticmethod
    def directory_handler(path_to_dirs, dirs, img_format):
        # If 'yolo' folder already exists, delete it 
        if os.path.exists(os.path.join(path_to_dirs, 'yolo')):
            shutil.rmtree(os.path.join(path_to_dirs, 'yolo'))

        # Iterating through 'dirs'
        for dir in dirs:
            dir_path = os.path.join(path_to_dirs, dir)
            output_path = os.path.join(path_to_dirs, 'yolo', dir)

            #if not os.path.exists(output_path):
            os.makedirs(output_path)

            # Obtaining all image paths from the directory
            image_basename_list = []
            for filename in glob.glob(os.path.join(dir_path, '*') + img_format):
                basename = os.path.basename(filename)
                basename_no_ext = os.path.splitext(basename)[0]
                image_basename_list.append(basename_no_ext)

            # Writing all the image paths in a .txt file
            with open(dir_path + '.txt', 'w') as image_basename_list_file:
                for basename_no_ext in image_basename_list:
                    image_basename_list_file.write(basename_no_ext + '\n')
                    ConvertVOCToYOLO.convert_annotation(basename_no_ext, dir_path, output_path)

            print("Converted " + dir_path)

        
        # Creating a yaml file consisting of class names
        with open(os.path.join(path_to_dirs, "classes.yaml"), "w") as f:
            yaml.dump({'classes': ConvertVOCToYOLO.classes}, f)


    @staticmethod
    def split_dataset(path_to_dirs, dirs, train_test_val_ratio, img_format, seed):
        if seed == None: seed = 0
        
        # Identifying which folder has images and which has XML files
        if dirs[0].lower().startswith('annot'):
            img_dir = 1
            xml_dir = 0
        elif dirs[1].lower().startswith('annot'):
            img_dir = 0
            xml_dir = 1

        # Obtaining the paths of all images
        path_list = glob.glob(os.path.join(path_to_dirs, dirs[img_dir], '*') + img_format)
        
        # Obtaining the basenames from the image paths
        basename_list = []
        for filename in path_list:
            basename = os.path.basename(filename)
            basename_no_ext = os.path.splitext(basename)[0]
            basename_list.append(basename_no_ext)

        # Setting a seed (default = 0) and shuffling the basenames
        random.seed(seed)
        random.shuffle(basename_list)

        # Splitting the basenames into train, test, and val folders according to the ratio given
        train_files, test_files, val_files = np.split(basename_list, [int(len(basename_list)*train_test_val_ratio[0]), int(len(basename_list)*(train_test_val_ratio[0] + train_test_val_ratio[1]))])

        # If 'split_data' folder already exists, delete it 
        if os.path.exists(os.path.join(path_to_dirs, 'split_data')):
            shutil.rmtree(os.path.join(path_to_dirs, 'split_data'))

        # Defining paths for the train, test and val folders
        train_path = os.path.join(path_to_dirs, 'split_data', 'train')
        test_path = os.path.join(path_to_dirs, 'split_data', 'test')
        val_path = os.path.join(path_to_dirs, 'split_data', 'val')
 
        # Creating train, test, val folders if the lists for the folders are non-empty
        if len(train_files) != 0: # and not os.path.exists(train_path):
            os.makedirs(train_path)
        if len(test_files) != 0: # and not os.path.exists(test_path):
            os.makedirs(test_path)
        if len(val_files) != 0: # and not os.path.exists(val_path):
            os.makedirs(val_path)

        # Copying images and XML files into the new train, text, val folders
        for filename in train_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename) + img_format, os.path.join(train_path, filename) + img_format)
            shutil.copy2(os.path.join(path_to_dirs, dirs[xml_dir], filename) + '.xml', os.path.join(train_path, filename) + '.xml')

        for filename in test_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename) + img_format, os.path.join(test_path, filename) + img_format)
            shutil.copy2(os.path.join(path_to_dirs, dirs[xml_dir], filename) + '.xml', os.path.join(test_path, filename) + '.xml')

        for filename in val_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename) + img_format, os.path.join(val_path, filename) + img_format)
            shutil.copy2(os.path.join(path_to_dirs, dirs[xml_dir], filename) + '.xml', os.path.join(val_path, filename) + '.xml')
        
        # Printing the number of images in each folder
        print('Number of images in train folder:', len(train_files))
        print('Number of images in test folder:', len(test_files))
        print('Number of images in val folder:', len(val_files))

        # Returning the folders that were created in order to convert it to YOLO
        if len(test_files) != 0 and len(val_files) != 0:
            return ['train', 'test', 'val']
        if len(test_files) != 0:
            return ['train', 'test']
        if len(val_files) != 0:
            return ['train', 'val']
        

    @staticmethod
    def convert(config_path):
        # Obtaining parameters from the config file
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            path_to_dirs = config['path_to_dirs']
            dirs = config['dirs']
            train_test_val_ratio = config['train_test_val_ratio']
            seed = config['seed']
            img_format = config['img_format']

        
        # Checking the 'dirs' list to identify the directory structure
        modified_dirs = [dir.lower() for dir in dirs]
        modified_dirs.sort()
        
        try:
            if modified_dirs[0].startswith('annot'):
                if len(modified_dirs) == 2 and (modified_dirs[1].startswith('image') or modified_dirs[1].startswith('img')):
                    new_dirs = ConvertVOCToYOLO.split_dataset(path_to_dirs, dirs, train_test_val_ratio, img_format, seed)
                    ConvertVOCToYOLO.directory_handler(os.path.join(path_to_dirs, 'split_data'), new_dirs, img_format)

                else:
                    raise Exception("Please provide 2 folders: 'annotations' and 'images'.")


            elif len(modified_dirs) == 2:
                if modified_dirs[0].startswith('test') and modified_dirs[1].startswith('train'):
                    ConvertVOCToYOLO.directory_handler(path_to_dirs, dirs, img_format)

                elif modified_dirs[0].startswith('train') and modified_dirs[1].startswith('val'):
                    ConvertVOCToYOLO.directory_handler(path_to_dirs, dirs, img_format)

                else:
                    raise Exception("Please provide either a 'test' folder or 'val' folder or both, along with the mandatory 'train' folder.")


            elif len(modified_dirs) == 3:
                if modified_dirs[0].startswith('test') and modified_dirs[1].startswith('train') and modified_dirs[2].startswith('val'):
                    ConvertVOCToYOLO.directory_handler(path_to_dirs, dirs, img_format)

                else:
                    raise Exception("Please provide either a 'test' folder or 'val' folder or both, along with the mandatory 'train' folder.")


            else:
                raise Exception("The following directory structures are accepted: \n(1) train, test, val \n(2) train, test \n(3) train, val \n(4) annotations, images")


        except Exception as e:
            print(e)


# If the script is run from the terminal
if __name__ == '__main__':
    try:
        arguments, values = getopt.getopt(sys.argv[1:], 'c:', ['config_path='])

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-c", "--config_path"):
                config_path = currentValue

        ConvertVOCToYOLO.convert(config_path)

    except getopt.GetoptError as err:
        print(str(err))     