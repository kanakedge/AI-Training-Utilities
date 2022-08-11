import os
from conversion_scripts.yolo_utils import yolo_2_imgs
from conversion_scripts.coco_utils import get_coco_annotations
from conversion_scripts.coco_to_voc import coco2voc
if __name__ == "__main__":
    # folder = "../wildfire_data/yolo_darknet_format/train"
    # files = os.listdir(folder)
    # img_files = []
    # for file in files:
    #     if file.endswith('.jpg'):
    #         img_file = os.path.join(folder, file)
    #         img_files.append(img_file)
    #
    #
    # print(get_coco_annotations(yolo_2_imgs(img_files[:10])))

    COCO_JSON = "_annotations.coco.json"
    OUTPUT = "xml_files"
    coco2voc(COCO_JSON, OUTPUT)