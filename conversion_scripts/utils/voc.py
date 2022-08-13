import xmltodict
import os

from conversion_scripts.utils.commons import read_img, get_label_name
from conversion_scripts.utils.yolo import read_yolo_txt, anno_yolo_voc

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
    "object": [{
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
    }]
}}


def read_voc_xml(path):
    with open(path) as fd:
        doc = xmltodict.parse(fd.read())
    bbox_objects = []
    if isinstance(doc['annotation']['object'], list):
        for obj in doc['annotation']['object']:
            bbox_objects.append(obj)
    if isinstance(doc['annotation']['object'], dict):
        bbox_objects.append(doc['annotation']['object'])
    return {'filename': doc['annotation']['filename'], "path": doc['annotation']['path'],
            "width": float(doc['annotation']['size']['width']), "height": float(doc['annotation']['size']['height']),
            "channels": float(doc['annotation']['size']['depth']), "bbox_objects": bbox_objects}


def create_xml(updated_details, output_folder=None, xml_structure=None):
    if xml_structure is None:
        xml_structure = XML_STRUCTURE
    filename = os.path.splitext(os.path.basename(updated_details["filename"]))[0] + ".xml"

    file_path = os.path.join(output_folder, filename)
    xml_structure['annotation']["folder"] = updated_details.get("folder", output_folder)
    xml_structure['annotation']["filename"] = filename
    xml_structure['annotation']["path"] = updated_details.get("path", None)
    xml_structure['annotation']["source"]["database"] = updated_details.get("database", None)
    xml_structure['annotation']["size"]["width"] = updated_details["width"]
    xml_structure['annotation']["size"]["height"] = updated_details["height"]
    xml_structure['annotation']["size"]["depth"] = updated_details["channels"]
    xml_structure['annotation']["segmented"] = updated_details.get("segmented", 0)

    objects = []
    for bbox in updated_details["bbox"]:
        objects.append(
            {"name": bbox["name"], "pose": bbox.get("pose", "Unspecified"), "truncated": bbox.get("truncated", 0),
             "difficult": bbox.get("difficult", 0), "occluded": bbox.get("occluded", 0),
             "bndbox": {"xmin": bbox["xmin"], "ymin": bbox["ymin"],
                        "xmax": bbox["xmax"], "ymax": bbox["ymax"]}
             }
        )

    xml_structure['annotation']["object"] = objects
    with open(file_path, "w") as f:
        f.write(xmltodict.unparse(xml_structure, pretty=True))


# def yolo_to_voc(txt_path, xml_folder, LABELS, img_folder=None):
#     meta_info = {}
#
#     if img_folder is None:
#         img_folder = os.path.dirname(txt_path)
#     basename = os.path.basename(txt_path)
#     file_no_ext = os.path.splitext(basename)[0]
#
#     meta_info['folder'] = xml_folder
#     meta_info['filename'] = file_no_ext + ".xml"
#     meta_info['path'] = os.path.join(xml_folder, meta_info['filename'])
#
#     img_info = read_img(None, file_no_ext, img_folder, ",jpg")
#     meta_info.extend(img_info)
#
#     bbox = read_yolo_txt(txt_path)
#     meta_info['name'] = get_label_name(LABELS, bbox['label'])
#
#     voc_bbox = anno_yolo_voc(bbox["x_center_norm"], bbox["y_center_norm"], bbox["width_norm"], bbox["height_norm"],
#                              meta_info["width"], meta_info["height"])
#     meta_info.update(voc_bbox)
#     try:
#         create_xml(meta_info['path'], meta_info)
#     except Exception as e:
#         raise Exception("Exception occured while writing XML file", e)
#     else:
#         return f"{meta_info['path']} created!"


def anno_voc_coco(x_min, x_max, y_min, y_max):
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    area = bbox_width * bbox_height

    return {"x_top_left": x_min, "y_top_left": y_min,
            "bbox_width": bbox_width, "bbox_height": bbox_height,
            "area": area}


def anno_voc_yolo(xmin, xmax, ymin, ymax, img_width, img_height):
    x_mean_norm = (xmax + xmin) / img_width
    y_mean_norm = (ymin + ymax) / img_height
    width_norm, height_norm = (xmax - xmin) / img_width, (ymax - ymin) / img_height
    return [x_mean_norm, y_mean_norm, width_norm, height_norm]


if __name__ == "__main__":
    xml = xmltodict.parse(xmltodict.unparse(XML_STRUCTURE, pretty=True))
