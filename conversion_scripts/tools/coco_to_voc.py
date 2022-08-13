import os
import shutil
from queue import Queue
from threading import Thread
from conversion_scripts.utils.voc import create_xml
from conversion_scripts.utils.coco import anno_coco_voc, parse_coco


def helper_coco2voc(json_file, q):
    parsed = parse_coco(json_file)
    for info in parsed:
        converted_results = []
        for bbox in info['bbox']:
            bbox_voc = anno_coco_voc(*bbox["bbox"])
            bbox_voc['name'] = bbox["category"]["name"]
            converted_results.append(bbox_voc)
        info['bbox'] = converted_results
        q.put(info)
    q.put(None)


def helper_createXML(q, output_dir):
    while True:
        details = q.get()
        if details is None:
            q.put(None)
            break
        create_xml(details, output_dir)


def coco2voc(json_file, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    q = Queue()
    num_threads = os.cpu_count()
    Thread(target=helper_coco2voc, args=(json_file, q,)).start()

    for _ in range(num_threads):
        Thread(target=helper_createXML, args=(q, output_folder,)).start()


if __name__ == "__main__":
    COCO_JSON = "../_annotations.coco.json"
    coco2voc(COCO_JSON, "../xml_files")
