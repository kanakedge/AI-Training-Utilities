import os
import shutil
from threading import Thread
from conversion_scripts.utils.coco import parse_coco, anno_coco_yolo
from conversion_scripts.utils.yolo import write_yolo_txt
from conversion_scripts.utils.commons import write_label_file


def helper_coco2yolo(json_file):
    parsed, categories = parse_coco(json_file)
    category_list = [category['name'] for category in categories]

    for info in parsed:
        converted_results = []
        for bbox in info['bbox']:
            bbox_yolo = anno_coco_yolo(*bbox["bbox"], info['width'], info['height'])
            label_id = bbox["category"]["id"]
            bbox_yolo.insert(0, label_id)
            converted_results.append(bbox_yolo)
        info['bbox'] = converted_results
        # q.put(info)
    return parsed, category_list


def helper_write_yolo_txt(parsed, out_dir):
    for info in parsed:
        filename = info['filename']
        filename = os.path.splitext(os.path.basename(filename))[0] + ".txt"
        filename = os.path.join(out_dir, filename)
        write_yolo_txt(filename, info['bbox'])


def coco2yolo(json_file, outdir):
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    num_threads = os.cpu_count()
    parsed, category_list = helper_coco2yolo(json_file)
    label_file = os.path.join(outdir, "category.labels")
    write_label_file(label_file, category_list)
    divisions = len(parsed) // num_threads
    start, end = 0, 0
    for idx in range(num_threads):
        if start < len(parsed):
            end = (start + divisions) if idx + 1 < num_threads else len(parsed)
            Thread(target=helper_write_yolo_txt, args=(parsed[start:end], outdir,)).start()
            start = start + divisions
