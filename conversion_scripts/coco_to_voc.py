import json
import os
import shutil
from queue import Queue
from threading import Thread

from conversion_scripts.anno_conversion import anno_coco_voc
from conversion_scripts.voc_utils import create_xml


def parse_coco(json_file, q: Queue):
    """
    json_file : coco.json
    q: queue.Queue()
    """
    with open(json_file) as f:
        data = json.load(f)
    categories = data["categories"]
    categories = {categories[idx]['id']: categories[idx] for idx in range(len(categories))}
    for img in data['images']:
        converted_results = []
        details = {}
        img_file = img["file_name"]
        width, height = img["width"], img["height"]
        for anno in data['annotations']:
            if anno['image_id'] == img["id"]:
                label = categories[anno['category_id']]["name"]
                xmin, ymin, bbox_width, bbox_height = anno["bbox"]
                bbox_voc = anno_coco_voc(xmin, ymin, bbox_width, bbox_height)
                bbox_voc["name"] = label
                converted_results.append(bbox_voc)
        details['filename'] = img_file
        details['width'], details["height"], details["channels"] = width, height, 3  # channels
        details["bbox"] = converted_results
        q.put(details)


def coco2voc(json_file, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    q = Queue()
    num_threads = os.cpu_count() - 2
    parse_coco(json_file, q)

    print("Num Threads: ", num_threads)
    for _ in range(num_threads):
        Thread(target=create_xml, args=(q, output_folder,)).start()


if __name__ == "__main__":
    COCO_JSON = "../_annotations.coco.json"
    coco2voc(COCO_JSON, "../xml_files")