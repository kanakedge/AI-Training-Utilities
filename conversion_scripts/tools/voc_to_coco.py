from queue import Queue
from threading import Thread
import os
from pprint import pprint
from conversion_scripts.utils.voc import read_voc_xml
from conversion_scripts.utils.commons import read_label_file
from conversion_scripts.utils.coco import write_coco


def helper_readXML(img_files, in_dir, q):
    for img_file in img_files:
        pth = os.path.splitext(img_file)[0] + '.xml'
        pth = os.path.join(in_dir, pth)
        voc_data = read_voc_xml(pth)
        details = {'filename': voc_data['filename'], "width": float(voc_data['width']),
                   'height': float(voc_data['height']), 'channels': int(voc_data['channels'])}
        bbox = []
        for obj in voc_data['bbox_objects']:
            bbox.append({"name": obj["name"], "xmin": float(obj['bndbox']['xmin']),
                         "xmax": float(obj['bndbox']['xmax']), "ymin": float(obj['bndbox']['ymin']),
                         "ymax": float(obj['bndbox']['ymax'])})
        details["bbox"] = bbox
        q.put(details)
    q.put(None)


def voc2coco(in_dir, img_dir, label_file, json_file):
    q = Queue()
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count()
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
            Thread(target=helper_readXML, args=(img_files[start:end], in_dir, q)).start()
            start = start + divisions
    Thread(target=write_coco, args=(labels, json_file, q, num_threads)).start()
