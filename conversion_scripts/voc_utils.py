import xmltodict
import os
from queue import Queue

XML_STRUCTURE = {'annotation': {
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
    parsed_ = {'filename': doc['annotation']['filename'], "path": doc['annotation']['path'],
               "img_width": doc['annotation']['size']['width'], "img_height": doc['annotation']['size']['height'],
               "img_depth": doc['annotation']['size']['depth'], "label": doc['annotation']['object']['name'],
               "x_top_left": (doc['annotation']['object']['bndbox'].values())[0],
               "x_bottom_right": (doc['annotation']['object']['bndbox'].values())[1],
               "y_top_left": (doc['annotation']['object']['bndbox'].values())[2],
               "y_bottom_right": (doc['annotation']['object']['bndbox'].values())[3]}

    return parsed_


def create_xml(q:Queue, output_folder=None, XML_STRUCTURE=XML_STRUCTURE):
    while not q.empty():
        updated_details = q.get()
        filename = os.path.splitext(updated_details["filename"])[0] + ".xml"
        if output_folder is None:
            output_folder = os.getcwd()
        file_path = os.path.join(output_folder, filename)
        XML_STRUCTURE['annotation']["folder"] = updated_details.get("folder", None)
        XML_STRUCTURE['annotation']["filename"] = filename
        XML_STRUCTURE['annotation']["path"] = updated_details.get("path", None)
        XML_STRUCTURE['annotation']["source"]["database"] = updated_details.get("database", None)
        XML_STRUCTURE['annotation']["size"]["width"] = updated_details["width"]
        XML_STRUCTURE['annotation']["size"]["height"] = updated_details["height"]
        XML_STRUCTURE['annotation']["size"]["depth"] = updated_details["channels"]
        XML_STRUCTURE['annotation']["segmented"] = updated_details.get("segmented", None)

        objects = []
        for bbox in updated_details["bbox"]:
            objects.append({"name": bbox["name"], "xmin": bbox["xmin"], "ymin": bbox["ymin"],
                           "xmax": bbox["xmax"], "ymax": bbox["ymax"], "pose": bbox.get("pose", None),
                           "truncated": bbox.get("truncated", None), "difficult": bbox.get("difficult", None)})

        XML_STRUCTURE['annotation']["object"] = objects
        with open(file_path, "w") as f:
            f.write(xmltodict.unparse(XML_STRUCTURE, pretty=True))


def yolo_to_voc(txt_path, xml_folder, LABELS, img_folder=None):
    meta_info = {}

    if img_folder is None:
        img_folder = os.path.dirname(txt_path)
    basename = os.path.basename(txt_path)
    file_no_ext = os.path.splitext(basename)[0]

    meta_info['folder'] = xml_folder
    meta_info['filename'] = file_no_ext + ".xml"
    meta_info['path'] = os.path.join(xml_folder, meta_info['filename'])

    img_info = read_img(None, file_no_ext, img_folder, ",jpg")
    meta_info.extend(img_info)

    bbox = read_yolo_txt(txt_path)
    meta_info['name'] = get_label_name(LABELS, bbox['label'])

    voc_bbox = anno_yolo_voc(bbox["x_center_norm"], bbox["y_center_norm"], bbox["width_norm"], bbox["height_norm"],
                             meta_info["width"], meta_info["height"])
    meta_info.update(voc_bbox)
    try:
        create_xml(meta_info['path'], meta_info)
    except Exception as e:
        print("Exception occured while writing XML file", e)
        return "Conversion Failed"
    else:
        return f"{meta_info['path']} created!"


if __name__ == "__main__":
    xml = xmltodict.parse(xmltodict.unparse(XML_STRUCTURE, pretty=True))
    print(xml)
