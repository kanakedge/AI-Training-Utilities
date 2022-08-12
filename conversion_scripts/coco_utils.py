import os
import json
from queue import Queue

from conversion_scripts.anno_conversion import anno_voc_coco


def create_json_categories(LABELS):
    categories = []
    for idx, label in enumerate(LABELS):
        categories.append({
            "id": idx,
            "name": label,
            "supercategory": None
        })
    return categories


def create_image_annotation(img_info):
    return {
        "file_name": os.path.basename(img_info["img_path"]),
        "height": img_info["height"],
        "width": img_info["width"],
        "id": img_info["id"],
    }


def create_coco_bbox_annotation(id, image_info, bbox_info, is_crowd=None):
    return {
        "id": id,
        "image_id": image_info["id"],
        "category_id": bbox_info["category_id"],
        "bbox": list(bbox_info.values())[1:-1],
        "area": bbox_info["area"],
        "is_crowd": is_crowd
    }


def get_coco_annotations(image_info_list):
    annotations = []
    for image in image_info_list:
        annotations.append(create_image_annotation(img_info=image))
    return annotations


def write_coco(labels, json_file, q: Queue):
    categories = create_json_categories(labels)
    json_data = {"categories": categories}
    images = []
    annotations = []
    img_id = 0
    bbox_id = 0
    while True:
        if not q.empty():
            details = q.get()
            if details is None:
                break
            images.append({"id": img_id, "license": None, "filename": os.path.basename(details['filename']),
                           "width": details["width"], "height": details["height"]})

            for bbox in details['bbox']:
                coco_bbox = anno_voc_coco(bbox['xmin'], bbox['xmax'], bbox['ymin'], bbox['ymax'])
                category_id = 0
                area = coco_bbox["area"]
                bbox_ = [coco_bbox["x_top_left"], coco_bbox["y_top_left"],
                         coco_bbox["bbox_width"], coco_bbox["bbox_height"]]
                is_crowd = None
                annotations.append({"id": bbox_id, "image_id": img_id, "category_id": category_id,
                                    "bbox": bbox_, "area": area, "segmentation": [], "iscrowd": is_crowd})
                bbox_id += 1

            img_id += 1

    json_data["images"] = images
    json_data["annotations"] = annotations
    with open(json_file, "w", encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
