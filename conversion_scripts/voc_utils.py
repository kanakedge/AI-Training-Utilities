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

def create_xml(file_path, updated_details, XML_STRUCTURE=XML_STRUCTURE):
    XML_STRUCTURE['annotation']["folder"] = updated_details.get("folder", None)
    XML_STRUCTURE['annotation']["filename"] = updated_details.get("filename")
    XML_STRUCTURE['annotation']["path"] = updated_details.get("path", None)
    XML_STRUCTURE['annotation']["source"]["database"] = updated_details.get("database", None)
    XML_STRUCTURE['annotation']["size"]["width"] = updated_details["width"]
    XML_STRUCTURE['annotation']["size"]["height"] = updated_details["height"]
    XML_STRUCTURE['annotation']["size"]["depth"] = updated_details["channels"]
    XML_STRUCTURE['annotation']["segmented"] = updated_details.get("segmented", None)
    XML_STRUCTURE['annotation']["object"]["name"] = updated_details["name"]
    XML_STRUCTURE['annotation']["object"]["pose"] = updated_details.get("pose", None)
    XML_STRUCTURE['annotation']["object"]["trucated"] = updated_details.get("truncated", None)
    XML_STRUCTURE['annotation']["object"]["difficult"] = updated_details.get("difficult", None)
    XML_STRUCTURE['annotation']["object"]["bndbox"]["xmin"] = updated_details["x_top_left"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["xmax"] = updated_details["x_bottom_right"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["ymin"] = updated_details["y_top_left"]
    XML_STRUCTURE['annotation']["object"]["bndbox"]["ymax"] = updated_details["y_bottom_right"]
    
    with open(file_path, "w") as f:
        f.write(xmltodict.unparse(XML_STRUCTURE, pretty= True))


def yolo_to_voc(txt_path, xml_folder, LABELS, img_folder = None):
    meta_info = {}

    if img_folder is None:
        img_folder = os.path.dirname(txt_path)
    basename = os.path.basename(txt_path)
    file_no_ext = os.path.splitext(basename)[0]
    
    meta_info['folder'] = xml_folder
    meta_info['filename'] = file_no_ext+".xml"
    meta_info['path'] = os.path.join(xml_folder, meta_info['filename'])
    
    img_info = read_img(None, file_no_ext, img_folder, ",jpg")
    meta_info.extend(img_info)

    bbox = read_yolo_txt(txt_path)
    meta_info['name'] = get_label_name(LABELS, bbox['label'])

    voc_bbox = anno_yolo_voc(bbox["x_center_norm"], bbox["y_center_norm"], bbox["width_norm"], bbox["height_norm"], meta_info["width"], meta_info["height"])
    meta_info.update(voc_bbox)
    try:
        create_xml(meta_info['path'], meta_info)
    except Exception as e:
        print("Exception occured while writing XML file", e)
        return "Conversion Failed"
    else:
        return f"{meta_info['path']} created!"
