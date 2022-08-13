import os
from conversion_scripts.utils.commons import read_img


def read_yolo_txt(txt_path):
    if os.path.exists(txt_path):
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
    raise FileNotFoundError(txt_path)  # todo return None and handle in calling function


def read_yolo(img_file, in_dir, img_dir):
    txt = os.path.splitext(img_file)[0] + ".txt"

    txt_file = os.path.join(in_dir, txt)
    img_file = os.path.join(img_dir, img_file)

    img_info = read_img(img_file)
    bbox_yolo = read_yolo_txt(txt_file)

    return {'filename': img_file, 'width': img_info["width"], "height": img_info["height"],
            "channels": img_info["channels"], "bbox": bbox_yolo}


def anno_yolo_coco(label, x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    bbox_width = width_norm * img_width
    bbox_height = height_norm * img_height

    x_top_left = x_center - bbox_width / 2
    y_top_left = y_center - bbox_height / 2

    area = bbox_width * bbox_height

    return {"category_id": label, "x_top_left": x_top_left, "y_top_left": y_top_left,
            "bbox_width": bbox_width, "bbox_height": bbox_height,
            "area": area}


def anno_yolo_voc(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    bbox_width = width_norm * img_width
    bbox_height = height_norm * img_height

    x_top_left, x_bottom_right = x_center - bbox_width / 2, x_center + bbox_width / 2
    y_top_left, y_bottom_right = y_center - bbox_height / 2, y_center + bbox_height / 2

    return {
        "xmin": x_top_left, "ymin": y_top_left,
        "xmax": x_bottom_right, "ymax": y_bottom_right,
    }
