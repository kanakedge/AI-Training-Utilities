import json
import os
from queue import Queue
from threading import Thread

from anno_conversion import anno_coco_voc
from voc_utils import create_xml

def parse_coco(json_file, q:Queue):
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
        details['width'], details["height"], details["channels"] = width, height, 3 #channels
        details["bbox"] = converted_results
        q.put(details)
    
class Coco2Yolo:
    """
    Coco to Yolo Conversion Class
    """
    def __init__(self, coco_json):
        with open(coco_json) as f:
            data = json.load(f)
        self.images = data['images']
        self.annotations = data['annotations']
        categories = data["categories"]
        self.categories = {categories[idx]['id']: categories[idx] for idx in range(len(categories))}
    
if __name__ == "__main__":
    JSON_FILE = "../../wildfire_data/coco_format/train/_annotations.coco.json"
    q = Queue()
    num_threads = os.cpu_count()
    parse_coco(JSON_FILE, q)
    for _ in len()
    create_xml(updated_details = q.get(), output_folder = "./xml_folder")
