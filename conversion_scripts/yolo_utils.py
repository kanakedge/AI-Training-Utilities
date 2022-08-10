from conversion_scripts.utils import read_img

def read_yolo_labels(label_file):
    with open(label_file, "r") as f:
        data = f.read()
    data = data.split('\n')
    return data
def read_yolo_txt(path):
    with open(path, "r") as file:
        data = file.read()
    data = data.split(" ")
    
    return {"label": int(data[0]), "x_center_norm":float(data[1]),
            "y_center_norm": float(data[2]), "width_norm": float(data[3]), 
            "height_norm": float(data[4])}

def yolo_2_imgs(img_path_list):
    images = []
    for idx, pth in enumerate(img_path_list):
        images.append(read_img(idx, pth, None, None, None))
    return images