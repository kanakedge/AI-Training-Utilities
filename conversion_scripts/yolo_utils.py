import os.path


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
