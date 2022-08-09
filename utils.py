from fileinput import filename
import xml.etree.ElementTree as ET
import xmltodict

XML_STRUCTURE = {'annotation':{
    "folder": None,
    "filename": None,
    "path": None,
    "source": {"database": None},
    "size": {
        "width": None,
        "height": None,
        "depth": None,
    },
    "segmented": None,
    "object": {
        "name": None,
        "pose": None,
        "truncated": None,
        "difficult": None,
        "occluded": None,
        "bndbox": {
            "xmin": None,
            "xmax": None,
            "ymin": None,
            "ymax": None
        }
    }
}}


def read_voc_xml(path):
    with open(path) as fd:
        doc = xmltodict.parse(fd.read())
    parsed_ = {}
    parsed_['filename'] = doc['annotation']['filename']
    parsed_["path"] = doc['annotation']['path']
    parsed_["img_width"] = doc['annotation']['size']['width']
    parsed_["img_height"] = doc['annotation']['size']['height']
    parsed_["img_depth"] = doc['annotation']['size']['depth']

    parsed_["label"] = doc['annotation']['object']['name']
    parsed_["x_top_left"], parsed_["x_bottom_right"], parsed_["y_top_left"], parsed_["y_bottom_right"] = doc['annotation']['object']['bndbox'].values()

    return parsed_


def read_yolo(path):
    with open(path, "r") as file:
        data = file.read()
    data = data.split(" ")
    
    return {"label": int(data[0]), "x_center":float(data[1]),
            "y_center": float(data[2]), "width": float(data[3]), 
            "height": float(data[4])}


demo_xml = {"folder", "filename", "database", "path", "width", "height", "depth", "segmented", "label", "pose", "truncated", "difficult", "occluded", "x_top_left", "y_top_left", "x_bottom_right", "y_bottom_right"}

def create_xml(file_path, updated_details, XML_STRUCTURE=XML_STRUCTURE):
    XML_STRUCTURE['annotation']["folder"] = updated_details.get("folder", None)
    XML_STRUCTURE['annotation']["filename"] = updated_details.get("filename", None)
    XML_STRUCTURE['annotation']["path"] = updated_details.get("path", None)
    XML_STRUCTURE['annotation']["source"]["database"] = updated_details.get("database", None)
    XML_STRUCTURE['annotation']["size"]["width"] = updated_details["width"]
    XML_STRUCTURE['annotation']["size"]["height"] = updated_details["height"]
    XML_STRUCTURE['annotation']["size"]["depth"] = updated_details["depth"]
    XML_STRUCTURE['annotation']["segmented"] = updated_details.get("segmented", None)
    XML_STRUCTURE['annotation']["object"]["name"] = updated_details["label"]
    XML_STRUCTURE['annotation']["object"]["pose"] = updated_details.get("pose", None)
    XML_STRUCTURE['annotation']["object"]["trucated"] = updated_details.get("truncated", None)
    XML_STRUCTURE['annotation']["object"]["difficult"] = updated_details("difficult", None)
    XML_STRUCTURE['annotation']["object"]["bndbox"]["xmin"] = updated_details["x_top_left"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["xmax"] = updated_details["x_bottom_right"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["ymin"] = updated_details["y_top_left"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["ymax"] = updated_details["y_bottom_right"]
    
    with open(file_path, "w") as f:
        f.write(xmltodict.unparse(XML_STRUCTURE, pretty= True))


if __name__ == "__main__":
    xml_path = "../wildfire_data/pascal_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf.aa2243c64fd18ad4e8c179d09c12cbfc.xml"
    txt_path = "../wildfire_data/yolo_darknet_format/train/ck0k9dg0vjxcg0848rmqzl38w_jpeg.rf.aa2243c64fd18ad4e8c179d09c12cbfc.txt"
    # print(read_voc_xml(path))
    # print(read_yolo(txt_path))
    create_xml()
    