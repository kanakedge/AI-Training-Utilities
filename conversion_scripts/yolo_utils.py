import os
from queue import Queue
from conversion_scripts.utils import read_img
from conversion_scripts.anno_conversion import anno_yolo_voc


def read_yolo_labels(label_file):
    with open(label_file, "r") as f:
        data = f.read()
    data = data.split('\n')
    return data


def read_yolo_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"{txt_path} not found")

    with open(txt_path, "r") as file:
        data = file.read()
    data = data.split("\n")
    bbox_ = []
    for txt in data:
        bbox = txt.split(" ")
        bbox_.append(
            {"label": int(bbox[0]), "x_center_norm": float(bbox[1]),
             "y_center_norm": float(bbox[2]), "width_norm": float(bbox[3]),
             "height_norm": float(bbox[4])}
        )
    return bbox_


def yolo_2_imgs(img_path_list):
    images = []
    for idx, pth in enumerate(img_path_list):
        images.append(read_img(idx, pth, None, None, None))
    return images


def read_yolo(img_files, labels, in_dir, img_dir, q: Queue):
    for idx, img_file in enumerate(img_files):
        details = {}

        txt = os.path.splitext(img_file)[0] + ".txt"

        txt_file = os.path.join(in_dir, txt)
        img_file = os.path.join(img_dir, img_file)

        img_info = read_img(idx, img_file)
        bbox_yolo = read_yolo_txt(txt_file)

        bbox_voc = []
        for bbox in bbox_yolo:
            box = anno_yolo_voc(bbox["x_center_norm"], bbox["y_center_norm"], bbox["width_norm"],
                                bbox["height_norm"], img_info["width"], img_info["height"])
            box["name"] = labels[bbox["label"]]
            bbox_voc.append(box)

        details['filename'] = img_file
        details['width'], details["height"], details["channels"] = \
            img_info["width"], img_info["height"], img_info["channels"]
        details["bbox"] = bbox_voc
        q.put(details)
    q.put(None)
