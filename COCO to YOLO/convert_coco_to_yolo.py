"""
Usage:

(1) From Terminal:
python convert_coco_to_yolo.py -c <config_file_path>
OR
python convert_coco_to_yolo.py --config_path <config_file_path>

(2) Importing:
from convert_coco_to_yolo import ConvertCOCOToYOLO
ConvertCOCOToYOLO.convert(<config_file_path>)
"""

import os
import glob
import json
import yaml
import sys
import getopt
import random
import numpy as np
import shutil

class ConvertCOCOToYOLO:
    @staticmethod
    def convert_annotation(json_file, output_path):
        with open(json_file) as f:
            data = json.load(f)
        
        images = data['images']
        annotations = data['annotations']

        for i in range(len(images)):
            converted_results = []
            for ann in annotations:
                if ann['image_id'] == images[i]['id']:
                    cat_id = int(ann['category_id'])
                    height = images[i]['height']
                    width = images[i]['width']
                    left, top, bbox_width, bbox_height = map(float, ann['bbox'])
                    cat_id -= 1
                    x_center, y_center = (left + bbox_width / 2, top + bbox_height / 2)
                    x_rel, y_rel = (x_center / width, y_center / height)
                    w_rel, h_rel = (bbox_width / width, bbox_height / height)
                    converted_results.append((cat_id, x_rel, y_rel, w_rel, h_rel))
                    image_name = images[i]['file_name']
                    basename = os.path.basename(image_name)
                    basename_no_ext = os.path.splitext(basename)[0]
                    with open(os.path.join(output_path, str(basename_no_ext)) + '.txt', 'w') as out_file:
                        out_file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))


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
            
            json_file = glob.glob(os.path.join(dir_path, '*') + '.json')
            ConvertCOCOToYOLO.convert_annotation(json_file[0], output_path)

            print("Converted " + dir_path)


    @staticmethod
    def split_dataset(path_to_dirs, dirs, train_test_val_ratio, img_format, seed):
        if seed == None: seed = 0
        # Identifying which folder has images and which has JSON files
        if dirs[0].lower().startswith('annot'):
            img_dir = 1
            json_dir = 0
        elif dirs[1].lower().startswith('annot'):
            img_dir = 0
            json_dir = 1

        # Obtaining the paths of all images
        path_list = glob.glob(os.path.join(path_to_dirs, dirs[img_dir], '*') + img_format)
        
        # Obtaining the basenames from the image paths
        image_list = []
        for filename in path_list:
            basename = os.path.basename(filename)
            image_list.append(basename)

        # Setting a seed (default = 0) and shuffling the basenames
        random.seed(seed)
        random.shuffle(image_list)

        # Splitting the basenames into train, test, and val folders according to the ratio given
        train_files, test_files, val_files = np.split(image_list, [int(len(image_list)*train_test_val_ratio[0]), int(len(image_list)*(train_test_val_ratio[0] + train_test_val_ratio[1]))])
        
        # If 'split_data' folder already exists, delete it 
        if os.path.exists(os.path.join(path_to_dirs, 'split_data')):
            shutil.rmtree(os.path.join(path_to_dirs, 'split_data'))
        
        # Defining paths for the train, test and val folders
        train_path = os.path.join(path_to_dirs, 'split_data', 'train')
        test_path = os.path.join(path_to_dirs, 'split_data', 'test')
        val_path = os.path.join(path_to_dirs, 'split_data', 'val')
 
        # Creating train, test, val folders if the lists for the folders are non-empty
        if len(train_files) != 0:
            os.makedirs(train_path)
        if len(test_files) != 0: 
            os.makedirs(test_path)
        if len(val_files) != 0:
            os.makedirs(val_path)
        
        # Reading the JSON file
        json_file = glob.glob(os.path.join(path_to_dirs, dirs[json_dir], '*') + '.json')[0]
        with open(json_file) as f:
            data = json.load(f)
        
        images = data['images']
        annotations = data['annotations']

        # Splitting the images and annotations fields in the JSON file for train, test, val
        train_images, train_images_id, test_images, test_images_id, val_images, val_images_id = [], [], [], [], [], []
        for dict in images:
            if dict['file_name'] in train_files:
                train_images.append(dict)
                train_images_id.append(dict['id'])
            elif dict['file_name'] in test_files:
                test_images.append(dict)
                test_images_id.append(dict['id'])
            elif dict['file_name'] in val_files:
                val_images.append(dict)
                val_images_id.append(dict['id'])

        train_annotations, test_annotations, val_annotations = [], [], []
        for dict in annotations:
            if dict['id'] in train_images_id:
                train_annotations.append(dict)
            elif dict['id'] in test_images_id:
                test_annotations.append(dict)
            elif dict['id'] in val_images_id:
                val_annotations.append(dict)


        # Copying images into train folder and creating the JSON file
        for filename in train_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename), os.path.join(train_path, filename))
        
        train_json_data = {
            'info' : data['info'],
            'licenses' : data['licenses'],
            'categories' :  data['categories'],
            'images' : train_images,
            'annotations' : train_annotations
        }

        with open(os.path.join(train_path, 'train') + '.json', 'w') as f:
            json.dump(train_json_data, f, indent = 4)

            
        # Copying images into test folder and creating the JSON file
        for filename in test_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename), os.path.join(test_path, filename))
            
        test_json_data = {
            'info' : data['info'],
            'licenses' : data['licenses'],
            'categories' :  data['categories'],
            'images' : test_images,
            'annotations' : test_annotations
        }

        with open(os.path.join(test_path, 'test') + '.json', 'w') as f:
            json.dump(test_json_data, f, indent = 4)

        # Copying images into val folder and creating the JSON file
        for filename in val_files:
            shutil.copy2(os.path.join(path_to_dirs, dirs[img_dir], filename), os.path.join(val_path, filename))
            
        val_json_data = {
            'info' : data['info'],
            'licenses' : data['licenses'],
            'categories' :  data['categories'],
            'images' : val_images,
            'annotations' : val_annotations
        }

        with open(os.path.join(val_path, 'val') + '.json', 'w') as f:
            json.dump(val_json_data, f, indent = 4)
        

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
                    new_dirs = ConvertCOCOToYOLO.split_dataset(path_to_dirs, dirs, train_test_val_ratio, img_format, seed)
                    ConvertCOCOToYOLO.directory_handler(os.path.join(path_to_dirs, 'split_data'), new_dirs, img_format)

                else:
                    raise Exception("Please provide 2 folders: 'annotations' and 'images'.")


            elif len(modified_dirs) == 2:
                if modified_dirs[0].startswith('test') and modified_dirs[1].startswith('train'):
                    ConvertCOCOToYOLO.directory_handler(path_to_dirs, dirs, img_format)

                elif modified_dirs[0].startswith('train') and modified_dirs[1].startswith('val'):
                    ConvertCOCOToYOLO.directory_handler(path_to_dirs, dirs, img_format)

                else:
                    raise Exception("Please provide either a 'test' folder or 'val' folder or both, along with the mandatory 'train' folder.")


            elif len(modified_dirs) == 3:
                if modified_dirs[0].startswith('test') and modified_dirs[1].startswith('train') and modified_dirs[2].startswith('val'):
                    ConvertCOCOToYOLO.directory_handler(path_to_dirs, dirs, img_format)

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

        ConvertCOCOToYOLO.convert(config_path)

    except getopt.GetoptError as err:
        print(str(err))     