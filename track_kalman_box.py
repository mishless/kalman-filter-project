import csv
from glob import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import matplotlib.patches as patches
from association.ml_associations_box import MLKalmanAssociation as DataAssociation
from kalman_filter_box import KalmanFilter
from ssd.ssd import SSDObjectDetection
from yolo.yolo_object_detection import YOLOObjectDetection
from scipy.stats import multivariate_normal
from find_datasets import get_iou
import pickle


frozen = "ssd/ssd_mobilenet_v1_coco_2017/frozen_inference_graph.pb"
pbtxt = "ssd/ssd_mobilenet_v1_coco_2017/config.pbtxt"
def main():
    R_values = [0.001, 0.1, 1, 10, 1000]
    Q_values = [0.001, 0.1, 1, 10, 1000]
    parser = argparse.ArgumentParser(
        description="Run SSD on input folder and show result in popup window")
    parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                        help="Specify which object detector network should be used")
    args = parser.parse_args()
    if args.object_detector == 'ssd':
        fd = SSDObjectDetection(frozen, pbtxt)
    elif args.object_detector == 'yolo_full':
        fd = YOLOObjectDetection('full')
    elif args.object_detector == 'yolo_tiny':
        fd = YOLOObjectDetection('tiny')
    should_plot = False

    # Get images list from dataset
    dataset = "data/TinyTLP/"
    for path, directories, files in os.walk(dataset):
        all_directories = directories
        break
    results = {}
    for l, R_value in enumerate(R_values):
        for k, Q_value in enumerate(Q_values):
            for dir in all_directories:
                iou = []
                results[dir] = []
                ground_truth_file = dataset + dir +"/groundtruth_rect.txt"
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

                initial_position = ground_truth[0]
                frame_id, x, y, w, h, is_lost = initial_position
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                kf = KalmanFilter(x=np.array([[x], [1], [y], [1], [x + w], [1], [y + h], [1]]), Q=Q_value, R=R_value)

                # Initialize features detector

                # Iterate of every image
                features = {}
                t = tqdm(images_filelist[1:], desc="Processing")
                da = DataAssociation(R=kf.R, H=kf.H, threshold=1)

                if should_plot:
                    fig, ax = plt.subplots(1)
                for i, im in enumerate(t):
                    img = plt.imread(images_filelist[i])
                    # Compute features
                    features[i] = np.array(fd.compute_features(img))

                    # Do prediction
                    mu_bar, Sigma_bar = kf.predict()

                    # Do data association
                    da.update_prediction(mu_bar, Sigma_bar)
                    m = da.associate(features[i])
                    kf.update(m)
                    gt = list(map(int, ground_truth[i]))
                    kf_x = kf.get_x()

                    if should_plot:
                        if m is not None:
                            feature_box = patches.Rectangle((m[0], m[1]), m[2] - m[0], m[3]- m[1], linewidth=1, linestyle="solid",
                                               edgecolor="r", facecolor=None, fill=None)
                            ax.add_patch(feature_box)

                        gt_box = patches.Rectangle((int(ground_truth[i][1]), int(ground_truth[i][2])), int(ground_truth[i][3]), int(ground_truth[i][4]), linewidth=1, linestyle="solid",
                                               edgecolor="b", facecolor=None, fill=None)
                        r = patches.Rectangle((kf_x[0], kf_x[1]), kf_x[2] - kf_x[0], kf_x[3] - kf_x[1], linewidth=1, linestyle="solid",
                                               edgecolor="y", facecolor=None, fill=None)
                        ax.add_patch(gt_box)
                        ax.add_patch(r)
                        plt.gca()
                        ax.imshow(img)
                        plt.pause(0.0001)
                        plt.cla()
                    iou.append(get_iou([kf_x[0], kf_x[1], kf_x[2], kf_x[3]], [gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                    results[dir].append([kf_x[0], kf_x[1], kf_x[2] - kf_x[0], kf_x[3] - kf_x[1]])
                print(f"Dataset: {dir}, IoU: {np.mean(iou), np.std(iou)}")
            with open(f"results/kalman_filter_box_R_{l}_Q_{k}_{args.object_detector}.pickle", 'wb') as fp:
                pickle.dump(results, fp)


if __name__ == '__main__':
    main()
