import os
import cv2


def read_label_file(label_file):
    with open(label_file, "r") as f:
        data = f.read()
    data = data.split('\n')
    return data


def get_label_name(labels, idx):
    return labels[idx]


def read_img(img_path=None, file_no_ext=None, img_folder=None, img_ext=".jpg"):
    if img_path is None:
        img_path = os.path.join(img_folder, file_no_ext + img_ext)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} missing!!")

    img = cv2.imread(img_path)
    shape = img.shape
    return {
        "img_path": img_path,
        "width": shape[1],
        "height": shape[0],
        "channels": shape[2]
    }


def write_label_file(label_file, labels):
    with open(label_file, "w") as f:
        f.writelines(f"{label}\n" for label in labels)


if __name__ == "__main__":
    xml_path = "../wildfire_data/pascal_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf" \
               ".aa2243c64fd18ad4e8c179d09c12cbfc.xml "
    txt_path = "../wildfire_data/yolo_darknet_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf" \
               ".aa2243c64fd18ad4e8c179d09c12cbfc.txt "
    # print(read_voc_xml(path))
    # print(read_yolo(txt_path))
    # LABELS = read_yolo_labels("../wildfire_data/yolo_darknet_format/train/_darknet.labels")
    # yolo_to_voc(txt_path, ".", LABELS)

    # print(voc_to_json(xml_path, ".", LABELS))
