import os
import shutil
from threading import Thread
from conversion_scripts.utils.voc import create_xml
from conversion_scripts.utils.coco import anno_coco_voc, parse_coco
from conversion_scripts.utils.commons import write_label_file


def helper_coco2voc(json_file):
    parsed, categories = parse_coco(json_file)
    category_list = [category['name'] for category in categories]
    for info in parsed:
        converted_results = []
        for bbox in info['bbox']:
            bbox_voc = anno_coco_voc(*bbox["bbox"])
            bbox_voc['name'] = bbox["category"]["name"]
            converted_results.append(bbox_voc)
        info['bbox'] = converted_results
    return parsed, category_list


def helper_createXML(parsed, output_dir):
    for info in parsed:
        create_xml(info, output_dir)


def coco2voc(json_file, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    parsed, category_list = helper_coco2voc(json_file)

    label_file = os.path.join(output_folder, "category.labels")
    write_label_file(label_file, category_list)

    num_threads = os.cpu_count()
    divisions = len(parsed) // num_threads
    start, end = 0, 0

    for idx in range(num_threads):
        if start < len(parsed):
            end = (start + divisions) if idx + 1 < num_threads else len(parsed)
            Thread(target=helper_createXML, args=(parsed[start:end], output_folder,)).start()
            start = start + divisions


if __name__ == "__main__":
    COCO_JSON = "../_annotations.coco.json"
    coco2voc(COCO_JSON, "../xml_files")
