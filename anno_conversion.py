"""
Yolo Darknet: (Label_ID |  X_Center_NORM | Y_Center_NORM | WIDTH_NORM | HEIGHT_NORM)
COCO: (x-top left, y-top left, width, height)
VOC: (xmin-top left, ymin-top left,xmax-bottom right, ymax-bottom right)
"""

def anno_yolo_coco(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    bbox_width = width_norm * img_width
    bbox_height = height_norm * img_height

    x_top_left = x_center - bbox_width/2
    y_top_left = y_center - bbox_height/2
    
    area = bbox_width * bbox_height

    return {"x_top_left":x_top_left, "y_top_left":y_top_left, 
            "bbox_width":bbox_width, "bbox_height":bbox_height,
            "area":area}
    

def anno_yolo_voc(x_center_norm, y_center_norm, width_norm, height_norm, img_width, img_height):
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    bbox_width = width_norm * img_width
    bbox_height = height_norm * img_height

    x_top_left, x_bottom_right = x_center - bbox_width/2, x_center + bbox_width/2 
    y_top_left, y_bottom_right = y_center - bbox_height/2, y_center + bbox_height/2 

    return {
        "x_top_left":x_top_left, "y_top_left": y_top_left, 
        "x_bottom_right": x_bottom_right, "y_bottom_right":y_bottom_right,
        "img_width":img_width, "img_height":img_height
     }

def anno_voc_coco(x_min, x_max, y_min, y_max):
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    area = bbox_width * bbox_height

    return {"x_top_left":x_min, "y_top_left":y_min, 
            "bbox_width":bbox_width, "bbox_height":bbox_height,
            "area":area}


    

def anno_coco_voc(x_top_left, y_top_left, bbox_width, bbox_height):
    x_bottom_right = x_top_left + bbox_width
    y_bottom_right = y_top_left + bbox_height

    return {
        "x_top_left":x_top_left, "y_top_left": y_top_left, 
        "x_bottom_right": x_bottom_right, "y_bottom_right":y_bottom_right,
    }
    


if __name__ == "__main__":
    coco_ex = {"height": 480,
            "width": 640,
            "bbox": [
                126,
                197,
                125,
                116
            ],
            'area': 14500
        }
    yolo_ex = {"x_center":0.29453125, "y_center":0.53125, "width":0.1953125, "height":0.24166666666666667}

    voc_ex = {"x_min":127, "x_max":252, "y_min":198, "y_max":314}

    print("COCO: ", anno_yolo_coco(yolo_ex["x_center"], yolo_ex["y_center"], yolo_ex["width"], yolo_ex["height"], 640, 480))
    print("VOC: ", anno_yolo_voc(yolo_ex["x_center"], yolo_ex["y_center"], yolo_ex["width"], yolo_ex["height"], 640, 480))
    print("COCO: ", anno_voc_coco(voc_ex["x_min"], voc_ex["x_max"], voc_ex["y_min"], voc_ex["y_max"]))
    print("VOC: ", anno_coco_voc(126, 197, 125, 116))