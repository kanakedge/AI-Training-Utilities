import os
import shutil
from threading import Thread
from conversion_scripts.utils.commons import read_label_file
from conversion_scripts.utils.voc import read_voc_xml, anno_voc_yolo
from conversion_scripts.utils.yolo import write_yolo_txt


def helper_voc2yolo(img_files, in_dir, labels_dict, out_dir, ):
    for img_file in img_files:
        xml_file = os.path.join(in_dir, os.path.splitext(img_file)[0] + ".xml")
        img_details = read_voc_xml(xml_file)
        converted_anno = []
        for bbox_obj in img_details["bbox_objects"]:
            label_id = labels_dict[bbox_obj["name"]]
            xmin, xmax, ymin, ymax = float(bbox_obj['bndbox']['xmin']), float(bbox_obj['bndbox']['xmax']), float(bbox_obj['bndbox']['ymin']), float(bbox_obj['bndbox']['ymax'])
            yolo_anno = anno_voc_yolo(xmin, xmax, ymin, ymax,
                                      img_details["width"], img_details["height"])
            yolo_anno.insert(0, label_id)
            converted_anno.append(yolo_anno)
        txt_file = os.path.join(out_dir, os.path.splitext(img_file)[0] + ".txt")
        write_yolo_txt(txt_file, converted_anno)


def voc2yolo(in_dir, img_dir, labels_file, out_dir):
    labels = read_label_file(labels_file)
    labels_dict = {}
    for idx, label in enumerate(labels):
        labels_dict[label] = idx

    img_extensions = [".png", ".jpg", ".jpeg"]
    img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[-1] in img_extensions]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    num_threads = os.cpu_count()
    divisions = len(img_files) // num_threads
    start, end = 0, 0

    for idx in range(num_threads):
        if start < len(img_files):
            end = (start + divisions) if idx + 1 < num_threads else len(img_files)
            Thread(target=helper_voc2yolo, args=(img_files[start:end],
                                                 in_dir, labels_dict, out_dir,)).start()
            start = start + divisions
