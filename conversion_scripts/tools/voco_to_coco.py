from queue import Queue
from threading import Thread
import os

from conversion_scripts.utils.voc import read_voc_xml


def voc2coco(in_dir, img_dir, json_file):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count()

    if img_dir is None:
        img_dir = in_dir
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[-1] in img_extensions]
    div = len(img_files) // num_threads
    start, end = 0, 0
    for _ in range(num_threads):
        Thread(target=read_voc_xml, args=(img_files[start:]))
