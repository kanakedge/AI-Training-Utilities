import xmltodict
import os
import cv2
from conversion_scripts.voc_utils import read_voc_xml
# demo_xml = {"folder", "filename", "database", "path", "width", "height", "channels", "segmented", "label", "pose", "truncated", "difficult", "occluded", "x_top_left", "y_top_left", "x_bottom_right", "y_bottom_right"}
# XML_STRUCTURE = {'annotation':{
#     "folder": None,
#     "filename": None,
#     "path": None,
#     "source": {"database": None},
#     "size": {
#         "width": None,
#         "height": None,
#         "depth": None,
#     },
#     "segmented": None,
#     "object": {
#         "name": None,
#         "pose": None,
#         "truncated": None,
#         "difficult": None,
#         "occluded": None,
#         "bndbox": {
#             "xmin": None,
#             "xmax": None,
#             "ymin": None,
#             "ymax": None
#         }
#     }
# }}


# def read_yolo_labels(label_file):
#     with open(label_file, "r") as f:
#         data = f.read()
#     data = data.split('\n')
#     return data

# def create_json_categories(LABELS):
#     categories = []
#     for idx, label in enumerate(LABELS):
#         categories.append({
#             "id": idx,
#             "name": label,
#             "supercategory": None
#         })
#     return categories


# def read_voc_xml(path):
#     with open(path) as fd:
#         doc = xmltodict.parse(fd.read())
#     parsed_ = {}
#     parsed_['filename'] = doc['annotation']['filename']
#     parsed_["path"] = doc['annotation']['path']
#     parsed_["img_width"] = doc['annotation']['size']['width']
#     parsed_["img_height"] = doc['annotation']['size']['height']
#     parsed_["img_depth"] = doc['annotation']['size']['depth']

#     parsed_["label"] = doc['annotation']['object']['name']
#     parsed_["x_top_left"], parsed_["x_bottom_right"], parsed_["y_top_left"], parsed_["y_bottom_right"] = doc['annotation']['object']['bndbox'].values()

#     return parsed_


# def read_yolo_txt(path):
#     with open(path, "r") as file:
#         data = file.read()
#     data = data.split(" ")
    
#     return {"label": int(data[0]), "x_center_norm":float(data[1]),
#             "y_center_norm": float(data[2]), "width_norm": float(data[3]), 
#             "height_norm": float(data[4])}

# def create_xml(file_path, updated_details, XML_STRUCTURE=XML_STRUCTURE):
#     XML_STRUCTURE['annotation']["folder"] = updated_details.get("folder", None)
#     XML_STRUCTURE['annotation']["filename"] = updated_details.get("filename")
#     XML_STRUCTURE['annotation']["path"] = updated_details.get("path", None)
#     XML_STRUCTURE['annotation']["source"]["database"] = updated_details.get("database", None)
#     XML_STRUCTURE['annotation']["size"]["width"] = updated_details["width"]
#     XML_STRUCTURE['annotation']["size"]["height"] = updated_details["height"]
#     XML_STRUCTURE['annotation']["size"]["depth"] = updated_details["channels"]
#     XML_STRUCTURE['annotation']["segmented"] = updated_details.get("segmented", None)
#     XML_STRUCTURE['annotation']["object"]["name"] = updated_details["name"]
#     XML_STRUCTURE['annotation']["object"]["pose"] = updated_details.get("pose", None)
#     XML_STRUCTURE['annotation']["object"]["trucated"] = updated_details.get("truncated", None)
#     XML_STRUCTURE['annotation']["object"]["difficult"] = updated_details.get("difficult", None)
#     XML_STRUCTURE['annotation']["object"]["bndbox"]["xmin"] = updated_details["x_top_left"]
#     XML_STRUCTURE['annotation']["object"]["bndbox"]["xmax"] = updated_details["x_bottom_right"]
#     XML_STRUCTURE['annotation']["object"]["bndbox"]["ymin"] = updated_details["y_top_left"]
#     XML_STRUCTURE['annotation']["object"]["bndbox"]["ymax"] = updated_details["y_bottom_right"]
    
#     with open(file_path, "w") as f:
#         f.write(xmltodict.unparse(XML_STRUCTURE, pretty= True))

def get_label_name(LABELS, idx):
    return LABELS[idx]

def read_img(idx, img_path=None, file_no_ext=None, img_folder=None, img_ext=".jpg"):
    if img_path is None:
        img_path = os.path.join(img_folder, file_no_ext+img_ext)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} missing!!")
        
    img = cv2.imread(img_path)
    shape = img.shape
    return {
        "id": idx,
        "img_path": img_path,
        "width": shape[0],
        "height": shape[1],
        "channels": shape[2]
    }


    
def voc_to_json(xml_path, LABELS):
    meta_info = read_voc_xml(xml_path)
    return meta_info









# def yolo_to_coco(txt_path, img_folder=None, img_ext='.jpg'):
#     bbox_yolo = read_yolo_txt(txt_path)
#     if not img_folder:
#         img_folder = os.path.dirname(txt_path)
#     txt_file = os.path.basename(txt_path)
#     file_no_ext = os.path.splitext(txt_file)[0]
#     img_info = read_img(None, file_no_ext, img_folder, ".jpg")


    
    
    anno_yolo_coco(bbox_yolo['x_center_norm'], bbox_yolo['y_center_norm'], bbox_yolo['width_norm'], bbox_yolo['height_norm'])

if __name__ == "__main__":
    xml_path = "../wildfire_data/pascal_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf.aa2243c64fd18ad4e8c179d09c12cbfc.xml"
    txt_path = "../wildfire_data/yolo_darknet_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf.aa2243c64fd18ad4e8c179d09c12cbfc.txt"
    # print(read_voc_xml(path))
    # print(read_yolo(txt_path))
    # LABELS = read_yolo_labels("../wildfire_data/yolo_darknet_format/train/_darknet.labels")
    # yolo_to_voc(txt_path, ".", LABELS)

    # print(voc_to_json(xml_path, ".", LABELS))
    
    