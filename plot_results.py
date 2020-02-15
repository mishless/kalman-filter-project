import numpy as np
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
from find_datasets import get_iou
import argparse

# Kalman Filter results with Tiny YOLO and varying R
R_values_index = [0, 1, 2, 3, 4]
R_values = [0.001, 0.1, 1, 10, 1000]
results = {}
keys = None
dataset = "data/TinyTLP/"
parser = argparse.ArgumentParser()
parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                    help="Specify which object detector was used in experiment")
args = parser.parse_args()
res = {}

for i, R_value in enumerate(R_values):
    with open(f"results/kalman_filter_box_R_{i}_Q_1_{args.object_detector}.pickle", 'rb') as f:
        results[R_value] = pickle.load(f)
        if keys is None:
            keys = list(results[R_value].keys())
    for key in keys:
        if key not in res:
            res[key] = []
        ground_truth_file = dataset + key + "/groundtruth_rect.txt"
        ground_truth = list(csv.reader(open(ground_truth_file)))
        current_res = results[R_value][key]

        zipped_res = list(zip(current_res,ground_truth))
        iou = list(map(lambda x: get_iou([int(x[0][0]), int(x[0][1]), int(x[0][0]) + int(x[0][2]), int(x[0][1]) + int(x[0][3])],
                                         [int(x[1][1]), int(x[1][2]), int(x[1][1]) + int(x[1][3]), int(x[1][2]) + int(x[1][4])]), zipped_res))
        res[key].append(iou)

for k, v in res.items():
    for i, v1 in enumerate(v):
        print(f"Dataset {k}, R={R_values[i]}, mean IoU {np.mean(v1)}, std {np.std(v1)}, number of frames the IoU is 0: {v1.count(0)}")
        plt.plot(v1, label=f"{i}")
    plt.legend()
    plt.show()
