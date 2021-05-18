import os
import glob
import xml.etree.ElementTree as ET

#### TO BE MODIFIED AS PER NEEDS ####
# Run this script from the folder that contains the 'dirs'
dirs = ['train', 'valid', 'test']  
classes = ['smoke']
img_format = '.jpg'
#####################################


def convert(img_w, img_h, xmin, xmax, ymin, ymax):
    # Converting VOC format to YOLO format
    x = (xmin + xmax)/2/img_w
    y = (ymin + ymax)/2/img_h
    w = (xmax - xmin)/img_w
    h = (ymax - ymin)/img_h

    return (x, y, w, h)


def convert_annotation(image_path, dir_path, output_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

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
            class_name = obj.find('name').text
            if class_name not in classes or difficult == 1:
                continue

            # From the class name given in the XML file, obtaining the class ID
            class_id = classes.index(class_name)

            # Obtaining xmin, xmax, ymin, ymax of the bounding box
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            # Converting to YOLO format
            annotation = convert(img_w, img_h, xmin, xmax, ymin, ymax)

            # Writing the YOLO format bounding box in a .txt file
            out_file.write(str(class_id) + " " + " ".join([str(val) for val in annotation]) + '\n')

# The 'dirs' must be in the folder where this script is
cwd = os.getcwd()

for dir in dirs:
    dir_path = os.path.join(cwd, dir)
    output_path = os.path.join(cwd, 'yolo', dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Obtaining all image paths from the directory
    image_path_list = []
    for filename in glob.glob(os.path.join(dir_path, '*') + img_format):
        image_path_list.append(filename)

    # Writing all the image paths in a .txt file
    with open(dir_path + '.txt', 'w') as image_path_list_file:
        for image_path in image_path_list:
            image_path_list_file.write(image_path + '\n')
            convert_annotation(image_path, dir_path, output_path)

    print("Converted " + dir_path)