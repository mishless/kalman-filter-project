import argparse
import os
import csv
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from constants import frozen, pbtxt
from ssd.ssd import SSDObjectDetection
from yolo.yolo_object_detection import YOLOObjectDetection


def rectify(array,
            bounds=None):
    if bounds is None:
        bounds = {0: (0, 100), 2: (0, 100)}
    for c, (b0, b1) in bounds.items():
        col = array[:, c]
        col[col < b0] = b0
        col[col > b1] = b1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run SSD on input folder and show result in popup window")
    parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                        help="Specify which object detector network should be used")
    parser.add_argument("test", choices=['Q', 'R'],
                        help="Which noise matrix should be tested")
    parser.add_argument("should_plot", default=False,
                        help="Whether or not to plot the boxes")
    args = parser.parse_args()
    return args


def load_object_detector(object_detector):
    if object_detector == 'ssd':
        fd = SSDObjectDetection(frozen, pbtxt)
    elif object_detector == 'yolo_full':
        fd = YOLOObjectDetection('full')
    elif object_detector == 'yolo_tiny':
        fd = YOLOObjectDetection('tiny')
    else:
        raise NotImplementedError
    return fd


def get_directories(dataset_path):
    for path, directories, files in os.walk(dataset_path):
        all_directories = directories
        break
    return all_directories


def extract_all_groud_truths(ground_truth_file):
    ground_truth = list(csv.reader(open(ground_truth_file)))
    gt_measurements = []
    for row in ground_truth:
        gt_measurements.append(np.array(
            [int(int(row[0]) - int(row[2]) / 2), int(int(row[1]) - int(row[3]) / 2)]))
    return ground_truth, gt_measurements


def create_box_kalman_filter(initial_position, Q, R):
    from kalman_filter_box import KalmanFilter
    # Extract initial position
    frame_id, x, y, w, h, is_lost = initial_position
    x, y, w, h = [float(i) for i in initial_position[1:5]]

    kf = KalmanFilter(x=np.array(
        [[x], [1], [y], [1], [x + w], [1], [y + h], [1]]), Q=Q, R=R)
    return kf


def plot_box_kf_result(ax, i, img, m, kf_x, ground_truth):
    if m is not None:
        feature_box = patches.Rectangle((m[0], m[1]), m[2] - m[0], m[3] - m[1], linewidth=1,
                                        linestyle="solid", edgecolor="r", facecolor=None, fill=None)
        ax.add_patch(feature_box)

    gt_box = patches.Rectangle((int(ground_truth[i][1]), int(ground_truth[i][2])), int(ground_truth[i][3]),
                               int(ground_truth[i][4]), linewidth=1, linestyle="solid",
                               edgecolor="b", facecolor=None, fill=None)
    r = patches.Rectangle((kf_x[0], kf_x[1]), kf_x[2] - kf_x[0], kf_x[3] - kf_x[1], linewidth=1, linestyle="solid",
                          edgecolor="y", facecolor=None, fill=None)
    ax.add_patch(gt_box)
    ax.add_patch(r)
    plt.gca()
    ax.imshow(img)
    plt.pause(0.0001)
    plt.cla()


def create_fix_ax():
    return plt.subplots(1)


def read_img(img_file):
    return plt.imread(img_file)
