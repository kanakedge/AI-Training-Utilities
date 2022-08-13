from queue import Queue
from threading import Thread
import os

from conversion_scripts.utils.yolo import read_yolo
from conversion_scripts.utils.yolo import read_yolo_labels
from conversion_scripts.utils.coco import write_coco


def yolo2coco(in_dir, img_dir, label_file, json_file=None):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count()
    labels = read_yolo_labels(label_file)

    if img_dir is None:
        img_dir = in_dir
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[-1] in img_extensions]
    for _ in range(num_threads):
        Thread(target=read_yolo, args=(img_files, labels, in_dir, img_dir, q,)).start()
    Thread(target=write_coco, args = (labels, json_file, q,)).start()

