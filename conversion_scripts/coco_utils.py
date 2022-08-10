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
def create_image_annotation(image_id, file_path, width, height):
    
    image_annotation = {
        "file_name": os.path.basename(file_path),
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation

def create_coco_bbox_annotation(id, image_id, category_id, bbox, area, is_crowd):
    return{
        "id": id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "is_crowd": is_crowd
    }

def get_coco_annotations(image_info_list):
    annotations = []
    for image in image_info_list:
        annotations.append(create_image_annotation(image['id'], image['img_path'], image['width'], image['height']))
    return annotations