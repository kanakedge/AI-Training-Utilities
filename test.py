from conversion_scripts.coco_to_voc import coco2voc
from conversion_scripts.yolo_to_voc import yolo2voc
from conversion_scripts.yolo_to_coco import yolo2coco
from conversion_scripts.voc_utils import read_voc_xml
if __name__ == "__main__":
    # COCO to VOC Conversion

    # COCO_JSON = "_annotations.coco.json"
    # OUTPUT = "xml_files"
    # coco2voc(COCO_JSON, OUTPUT)

    # YOLO to VOC Conversion

    # in_dir = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train"
    # label_file = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train/_darknet.labels"
    # out_dir = "./xml_files"
    # yolo2voc(in_dir, None, label_file, out_dir)

    # yolo2coco(in_dir, in_dir, label_file, "annotations.json")
    path = "ck0kcoc8ik6ni0848clxs0vif_jpeg.rf.8b4629777ffe1d349cc970ee8af59eac.xml"
    print(read_voc_xml(path))

