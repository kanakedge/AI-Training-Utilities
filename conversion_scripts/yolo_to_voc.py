import os
import shutil
from queue import Queue
from threading import Thread

from conversion_scripts.yolo_utils import read_yolo_labels, read_yolo
from conversion_scripts.voc_utils import create_xml


def yolo2voc(in_dir, img_dir, label_file, out_dir):
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