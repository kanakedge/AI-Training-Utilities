import os
import shutil
from tqdm import tqdm
from threading import Thread

from conversion_scripts.utils.yolo import read_yolo_labels, read_yolo
from conversion_scripts.utils.voc import create_xml


def helper_yolo2voc(img_files, labels, in_dir, img_dir, out_dir):
    for img_file in tqdm(img_files):
        details = read_yolo(img_file, labels, in_dir, img_dir)
        create_xml(details, output_folder=out_dir)


def yolo2voc(in_dir, img_dir, label_file, out_dir):
    img_extensions = [".jpg", ".jpeg", ".png"]
    num_threads = os.cpu_count()
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    labels = read_yolo_labels(label_file)

    if img_dir is None:
        img_dir = in_dir
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[-1] in img_extensions]

    divisions = len(img_files) // num_threads
    start, end = 0, 0
    for idx in range(num_threads):
        if start < len(img_files):
            end = (start + divisions) if idx + 1 < num_threads else len(img_files)
            Thread(target=helper_yolo2voc, args=(img_files[start:end],
                                                 labels, in_dir, img_dir, out_dir,)).start()
            start = start + divisions


if __name__ == "__main__":
    pass
