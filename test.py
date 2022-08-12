from conversion_scripts.coco_to_voc import coco2voc
from conversion_scripts.yolo_to_voc import yolo2voc

if __name__ == "__main__":
    # COCO to VOC Conversion

    # COCO_JSON = "_annotations.coco.json"
    # OUTPUT = "xml_files"
    # coco2voc(COCO_JSON, OUTPUT)

    # YOLO to VOC Conversion

    in_dir = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train"
    label_file = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train/_darknet.labels"
    out_dir = "./xml_files"
    yolo2voc(in_dir, None, out_dir, label_file)
