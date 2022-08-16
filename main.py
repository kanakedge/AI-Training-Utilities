import argparse


def get_arg_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Annotation Conversion (yolo, coco, voc)', add_help=add_help)
    parser.add_argument('current', type=str, help='coco, yolo, voc, anyone only')
    parser.add_argument('to', type=str, help='coco, yolo, voc, anyone only')
    parser.add_argument('-label_folder', type=str, help='folder with label files')
    parser.add_argument('-label_file', type=str, help='.labels file', required=False)
    parser.add_argument('-image_folder', type=str, help='folder with image files', required=False)
    parser.add_argument('-save_dir', type=str, help='folder to write annotations file', required=False)
    parser.add_argument('-coco_json', type=str, help='COCO Json file for conversion', required=False)
    parser.add_argument('-out_json', type=str, help='location to save json file', required=False)
    return parser


def main(args):
    if args.current.strip() == 'yolo' and args.to.strip() == 'coco':
        from conversion_scripts.tools.yolo_to_coco import yolo2coco
        if None in [args.LABEL_FOLDER, args.LABEL_FILE, args.OUT_JSON, args.IMAGE_FOLDER]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        yolo2coco(args.LABEL_FOLDER, args.IMAGE_FOLDER, args.LABEL_FILE, args.OUT_JSON)
    elif args.current.strip() == 'yolo' and args.to.strip() == 'voc':
        from conversion_scripts.tools.yolo_to_voc import yolo2voc
        if None in [args.LABEL_FOLDER, args.LABEL_FILE, args.SAVE_DIR, args.IMAGE_FOLDER]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        yolo2voc(args.LABEL_FOLDER, args.IMAGE_FOLDER, args.LABEL_FILE, args.SAVE_DIR)

    elif args.current.strip() == 'coco' and args.to.strip() == 'yolo':
        from conversion_scripts.tools.coco_to_yolo import coco2yolo
        if None in [args.COCO_JSON, args.SAVE_DIR]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        coco2yolo(args.COCO_JSON, args.SAVE_DIR)
    elif args.current.strip() == 'coco' and args.to.strip() == 'voc':
        from conversion_scripts.tools.coco_to_voc import coco2voc
        if None in [args.COCO_JSON, args.SAVE_DIR]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        coco2voc(args.COCO_JSON, args.SAVE_DIR)

    elif args.current.strip() == 'voc' and args.to.strip() == 'yolo':
        from conversion_scripts.tools.voc_to_yolo import voc2yolo
        if None in [args.LABEL_FOLDER, args.LABEL_FILE, args.IMAGE_FOLDER, args.SAVE_DIR]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        voc2yolo(args.LABEL_FOLDER, args.IMAGE_FOLDER, args.LABEL_FILE, args.SAVE_DIR)
    elif args.current.strip() == 'voc' and args.to.strip() == 'coco':
        from conversion_scripts.tools.voc_to_coco import voc2coco
        if None in [args.LABEL_FOLDER, args.LABEL_FILE, args.IMAGE_FOLDER, args.OUT_JSON]:
            raise InterruptedError('yolo2coco requires label_folder, image_folder, out_json and .labels file ')
        voc2coco(args.LABEL_FOLDER, args.IMAGE_FOLDER, args.LABEL_FILE, args.OUT_JSON)


class Arguments:
    current = 'voc'
    to = 'yolo'
    LABEL_FOLDER = '/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train' if current == 'yolo' else '/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/voc/train'
    LABEL_FILE = '/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train/_darknet.labels'
    IMAGE_FOLDER = LABEL_FOLDER
    OUT_JSON = './annotation_yolo.json' if current == 'yolo' else './annotation_xml.json'
    SAVE_DIR = "./yolo_files" if to == 'yolo' else './xml_files'
    COCO_JSON = '_annotations.coco.json'


if __name__ == "__main__":
    # args = get_arg_parser().parse_args()
    args = Arguments()
    main(args)
