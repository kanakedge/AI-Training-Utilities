import os
import shutil
import time
from queue import Queue
from threading import Thread

from conversion_scripts.yolo_utils import read_yolo_txt, read_yolo_labels
from conversion_scripts.utils import read_img
from conversion_scripts.anno_conversion import anno_yolo_voc
from conversion_scripts.voc_utils import create_xml


def read_yolo(img_files, labels, in_dir, img_dir, q: Queue):
    for idx, img_file in enumerate(img_files):
        details = {}

        txt = os.path.splitext(img_file)[0] + ".txt"

        txt_file = os.path.join(in_dir, txt)
        img_file = os.path.join(img_dir, img_file)

        img_info = read_img(idx, img_file)
        bbox_yolo = read_yolo_txt(txt_file)

        bbox_voc = []
        for bbox in bbox_yolo:
            box = anno_yolo_voc(bbox["x_center_norm"], bbox["y_center_norm"], bbox["width_norm"],
                                bbox["height_norm"], img_info["width"], img_info["height"])
            box["name"] = labels[bbox["label"]]
            bbox_voc.append(box)

        details['filename'] = img_file
        details['width'], details["height"], details["channels"] = \
            img_info["width"], img_info["height"], img_info["channels"]
        details["bbox"] = bbox_voc
        q.put(details)
    q.put(None)


def yolo2voc(in_dir, img_dir, out_dir, label_file):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    labels = read_yolo_labels(label_file)

    if img_dir is None:
        img_dir = in_dir
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[-1] in img_extensions]
    for _ in range(num_threads):
        Thread(target=read_yolo, args=(img_files, labels, in_dir, img_dir, q,)).start()
    for _ in range(num_threads):
        Thread(target=create_xml, args=(q, out_dir,)).start()


if __name__ == "__main__":
    pass
