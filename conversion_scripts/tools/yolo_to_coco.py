from queue import Queue
from threading import Thread
import os

from conversion_scripts.utils.yolo import read_yolo
from conversion_scripts.utils.yolo import read_yolo_labels
from conversion_scripts.utils.coco import write_coco


def helper_readYolo(img_files, labels, in_dir, img_dir, q):
    for img_file in img_files:
        q.put(read_yolo(img_file, labels, in_dir, img_dir))
    q.put(None)


def yolo2coco(in_dir, img_dir, label_file, json_file=None):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count() - 1
    labels = read_yolo_labels(label_file)

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
