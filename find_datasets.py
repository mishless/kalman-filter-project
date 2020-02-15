import argparse
import csv
import os
import pickle
from glob import glob

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ssd.ssd import SSDObjectDetection
from yolo.yolo_object_detection import YOLOObjectDetection


def get_iou(bb1, bb2):
    if bb1[0] > bb1[2] or bb1[1] > bb1[3] or bb2[0] > bb2[2] or bb2[1] > bb2[3]:
        print("Not a valid box detected")
        return 0

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def main():
    parser = argparse.ArgumentParser(
        description="Run SSD on input folder and show result in popup window")
    parser.add_argument("object_detector", choices=['ssd', 'yolo'],
                        help="Specify which object detector network should be used")
    args = parser.parse_args()
    if args.object_detector == 'ssd':
        fd = SSDObjectDetection(frozen, pbtxt)
    elif args.object_detector == 'yolo':
        fd = YOLOObjectDetection()

    # Config
    should_plot = False

    # Get images list from dataset
    dataset = "data/TinyTLP/"

    for path, directories, files in os.walk(dataset):
        all_directories = directories
        break
    results = {}
    all_directories = ['CarChase2', 'Mohiniyattam', 'Drone1']
    for dir in all_directories:
        results[dir] = []
        ground_truth_file = dataset + dir + "/groundtruth_rect.txt"
        images_wildcard = dataset + dir + "/img/*.jpg"
        images_filelist = glob(images_wildcard)

        # Sort them in ascending order
        # images_filelist = sorted(images_filelist, key=lambda xx: int(
        #     xx.split('/')[-1].split('.')[0]))

        # Extract all ground truths
        ground_truth = list(csv.reader(open(ground_truth_file)))
        gt_measurements = []
        for row in ground_truth:
            gt_measurements.append(np.array([int(int(row[0]) - int(row[2]) / 2), int(int(row[1]) - int(row[3]) / 2)]))
        # Iterate of every image
        features = {}
        t = tqdm(images_filelist[1:], desc="Processing")
        if should_plot:
            fig, ax = plt.subplots(1)
        for i, im in enumerate(t):
            img = plt.imread(images_filelist[i])
            height, width, _ = img.shape
            # Compute features
            features[i] = np.array(fd.compute_features(img))
            gt = list(map(int, ground_truth[i]))
            iou = 0.0
            for feature in features[i]:
                if should_plot:
                    feature_box = patches.Rectangle((feature[0], feature[1]), feature[2] - feature[0],
                                                    feature[3] - feature[1], linewidth=1,
                                                    linestyle="solid",
                                                    edgecolor="y", facecolor=None, fill=None)
                    ax.add_patch(feature_box)

                bbox1 = [gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]
                bbox2 = feature
                temp_iou = get_iou(bbox1, bbox2)
                if temp_iou > iou:
                    iou = temp_iou

            results[dir].append(iou)
            if should_plot:
                gt_box = patches.Rectangle((int(gt[1]), int(gt[2])), int(gt[3]), int(gt[4]), linewidth=1,
                                           linestyle="solid", edgecolor="b", facecolor=None, fill=None)
                ax.add_patch(gt_box)
                plt.gca()
                ax.imshow(img)
                plt.pause(0.0001)
                plt.cla()

        print(f"Dataset: {dir} mean iou: {np.mean(results[dir])}, std: {np.std(results[dir])}")
    with open(f"results/datasets_iou_full_yolo.pickle", 'wb') as fp:
        pickle.dump(results, fp)


if __name__ == '__main__':
    main()
