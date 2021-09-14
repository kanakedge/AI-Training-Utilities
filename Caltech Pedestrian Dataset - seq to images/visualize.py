import os
import glob
import json
import cv2 
import yaml
import sys
import getopt

class VisualizeVOCAndYOLOLabels:
    img_dir_path = None
    annot_dir_path = None
    img_format = None

    @staticmethod
    def visualizeBB(basenames):
        for name in basenames:
            image = cv2.imread(os.path.join(VisualizeVOCAndYOLOLabels.img_dir_path, name) + VisualizeVOCAndYOLOLabels.img_format)

            with open(os.path.join(VisualizeVOCAndYOLOLabels.annot_dir_path, name) + '.json') as file:
                data = json.load(file)
                
                for i in range(len(data)):
                    bbox = data[i]['pos']

                    xmin = bbox[0]
                    xmax = xmin + bbox[2]
                    ymin = bbox[1]
                    ymax = ymin + bbox[3]

                    print('Coordinates (xmin, xmax, ymin, ymax): ', xmin, xmax, ymin, ymax)

                    img = cv2.rectangle(image, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 1)
                    cv2.imshow('Image with Bounding Boxes', img)
                    cv2.waitKey(0)


            print()


    @staticmethod
    def visualize(config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            VisualizeVOCAndYOLOLabels.img_dir_path = config['img_dir_path']
            VisualizeVOCAndYOLOLabels.annot_dir_path = config['annot_dir_path']
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