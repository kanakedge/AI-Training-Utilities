import os

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
    return{
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
        annotations.append(create_image_annotation(image['id'], image['img_path'], image['width'], image['height']))
    return annotations