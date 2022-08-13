from queue import Queue
from threading import Thread
import os

from conversion_scripts.utils.yolo import read_yolo, anno_yolo_voc
from conversion_scripts.utils.coco import write_coco
from conversion_scripts.utils.commons import read_label_file


def helper_readYolo(img_files, labels, in_dir, img_dir, q):
    for img_file in img_files:
        details = read_yolo(img_file, in_dir, img_dir)

        bbox_voc = []
        for bbox in details["bbox"]:
            box = anno_yolo_voc(x_center_norm=bbox["x_center_norm"], y_center_norm=bbox["y_center_norm"],
                                height_norm=bbox["height_norm"], width_norm=bbox["width_norm"],
                                img_width=details["width"], img_height=details["height"])
            box["name"] = labels[bbox["label"]]
            bbox_voc.append(box)
        details["bbox"] = bbox_voc

        q.put(details)
    q.put(None)


def yolo2coco(in_dir, img_dir, label_file, json_file=None):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count() - 1
    labels = read_label_file(label_file)

    if img_dir is None:
        img_dir = in_dir
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[-1] in img_extensions]

    divisions = len(img_files) // num_threads
    start, end = 0, 0

    for idx in range(num_threads):
        if start < len(img_files):
            end = (start + divisions) if idx + 1 < num_threads else len(img_files)
            Thread(target=helper_readYolo, args=(img_files[start:end],
                                                 labels, in_dir, img_dir, q)).start()
            start = start + divisions

    Thread(target=write_coco, args=(labels, json_file, q, num_threads)).start()
