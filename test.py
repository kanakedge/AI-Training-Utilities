from conversion_scripts.tools.yolo_to_voc import yolo2voc
from conversion_scripts.tools.yolo_to_coco import yolo2coco
from conversion_scripts.tools.coco_to_voc import coco2voc
from conversion_scripts.tools.voc_to_coco import voc2coco
from conversion_scripts.utils.voc import read_voc_xml
from conversion_scripts.tools.voc_to_yolo import voc2yolo

import pprint

if __name__ == "__main__":
    # COCO to VOC Conversion
    #
    # COCO_JSON = "_annotations.coco.json"
    # OUTPUT = "xml_files"
    # coco2voc(COCO_JSON, OUTPUT)

    # YOLO to VOC Conversion

    # in_dir = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train"
    # label_file = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train/_darknet.labels"
    # out_dir = "./xml_files"
    # yolo2voc(in_dir, None, label_file, out_dir)

    # yolo2coco(in_dir, in_dir, label_file, "./yolo_annotations.json")
    #


    # path = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/voc/test/ck0kcoc8ik6ni0848clxs0vif_jpeg.rf.8b4629777ffe1d349cc970ee8af59eac.xml"
    # pprint.pprint(read_voc_xml(path))
    #
    in_dir = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/voc/train"
    label_file = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/voc/_darknet.labels"
    # voc2coco(in_dir, in_dir, label_file, "./xml_annotations.json")

    # VOC to YOLO
    voc2yolo(in_dir=in_dir, img_dir=in_dir, out_dir="yolo_files/", labels_file=label_file)

