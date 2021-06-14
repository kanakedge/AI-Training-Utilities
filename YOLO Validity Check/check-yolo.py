"""
Usage:

(1) From Terminal:
python check_yolo.py -c <config_file_path>
OR
python check_yolo.py --config_path <config_file_path>

(2) Importing:
from check_yolo.py import CheckYOLO
CheckYOLO.check(<config_file_path>)
"""

import os
import glob
import shutil
import yaml
import sys
import getopt

class CheckYOLO:

    @staticmethod
    def check(config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            yolo_dir_path = config['yolo_dir_path']


        invalid_files = []

        for filename in glob.glob(os.path.join(yolo_dir_path, '*') + '.txt'):
            with open(filename) as txt_file:
                line = txt_file.readline()
                while line:
                    class_id, x, y, w, h = map(float, line.split())

                    if x < 0 or y < 0 or w < 0 or h < 0:
                        invalid_files.append(os.path.basename(filename))
                        break

                    line = txt_file.readline()


        if len(invalid_files) == 0:
            print('No negative values found in the YOLO annotations.')
        else:
            if not os.path.exists(os.path.join(yolo_dir_path, 'yolo-invalid')):
                os.makedirs(os.path.join(yolo_dir_path, 'yolo-invalid'))
            
            for file in invalid_files:
                print('Negative values found in {}'.format(file))
                shutil.move(os.path.join(yolo_dir_path, file), os.path.join(yolo_dir_path, 'yolo-invalid', file))

            print('All invalid files have been moved to {}'.format(os.path.join(yolo_dir_path, 'yolo-invalid')))


# If the script is run from the terminal
if __name__ == '__main__':
    try:
        arguments, values = getopt.getopt(sys.argv[1:], 'c:', ['config_path='])

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-c", "--config_path"):
                config_path = currentValue

        CheckYOLO.check(config_path)

    except getopt.GetoptError as err:
        print(str(err))     