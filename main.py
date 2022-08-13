def main(conversion, label_folder, label_file, img_folder, out_folder):
    if conversion == "yolo2voc":
        from conversion_scripts.tools.yolo_to_voc import yolo2voc
        yolo2voc(in_dir=label_folder, img_dir=img_folder, label_file=label_file, out_dir=out_folder)


if __name__ == "__main__":
    conversion = ["yolo2voc"]
    in_dir = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train"
    label_file = "/Users/kanakraj/workspace/edgeneural/wild_fire_dataset/yolo/train/_darknet.labels"
    out_dir = "./xml_files"

    main(conversion[0], label_folder=in_dir, img_folder=in_dir, label_file=label_file, out_folder=out_dir)
