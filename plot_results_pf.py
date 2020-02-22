import numpy as np
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
import csv
from find_datasets import get_iou
import argparse

# Particle Filter results with Tiny YOLO and varying R
R_values = [0.001, 0.1, 1, 10, 1000]
Q_values = [0.001, 0.1, 1, 10, 1000]
results = {}
keys = None
dataset = "data/TinyTLP/"
parser = argparse.ArgumentParser()
parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                    help="Specify which object detector was used in experiment")
args = parser.parse_args()
res = {}
print("Particle Filter Box tracking with R values 0.001, 0.01, 1, 10, 1000")
for i, R_value in enumerate(R_values):
    j = 2
    Q_value = 1
    try:
        with open(f"results/particle_filter_box_R_{i}_Q_{j}_{args.object_detector}.pickle", 'rb') as f:
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
    except:
        pass

for k, v in res.items():
    print(f"{k}", end = ' & ')
    for i, v1 in enumerate(v):
        print(f"${format(np.mean(v1), '.2f')}\pm{format(np.std(v1), '.2f')}$", end = ' & ')
    print()
for k, v in res.items():
    print(f"{k}", end=' & ')
    for i, v1 in enumerate(v):
        print(f"{v1.count(0)}/600", end=' & ')
    print()


print("Particle Filter Box tracking with Q values 0.001, 0.01, 1, 10, 1000")
res = {}
for j, Q_value in enumerate(Q_values):
    i = 2
    R_value = 1
    try:
        with open(f"results/particle_filter_box_R_{i}_Q_{j}_{args.object_detector}.pickle", 'rb') as f:
            results[Q_value] = pickle.load(f)
            if keys is None:
                keys = list(results[Q_value].keys())
        for key in keys:
            if key not in res:
                res[key] = []
            ground_truth_file = dataset + key + "/groundtruth_rect.txt"
            ground_truth = list(csv.reader(open(ground_truth_file)))
            current_res = results[Q_value][key]

            zipped_res = list(zip(current_res,ground_truth))
            iou = list(map(lambda x: get_iou([int(x[0][0]), int(x[0][1]), int(x[0][0]) + int(x[0][2]), int(x[0][1]) + int(x[0][3])],
                                             [int(x[1][1]), int(x[1][2]), int(x[1][1]) + int(x[1][3]), int(x[1][2]) + int(x[1][4])]), zipped_res))
            res[key].append(iou)
    except:
        pass

for k, v in res.items():
    print(f"{k}", end = ' & ')
    for i, v1 in enumerate(v):
        print(f"${format(np.mean(v1), '.2f')}\pm{format(np.std(v1), '.2f')}$", end = ' & ')
    print()
for k, v in res.items():
    print(f"{k}", end = ' & ')
    for i, v1 in enumerate(v):
        print(f"{v1.count(0)}/600", end = ' & ')
    print()
res = {}
print("Particle Filter point tracking with R values 0.001, 0.01, 1, 10, 1000")
for i, R_value in enumerate(R_values):
    j = 2
    Q_value = 1
    try:
        with open(f"results/particle_filter_point_R_{i}_Q_{j}_{args.object_detector}.pickle", 'rb') as f:
            results[R_value] = pickle.load(f)
            if keys is None:
                keys = list(results[R_value].keys())
        for key in keys:
            if key not in res:
                res[key] = []
            ground_truth_file = dataset + key + "/groundtruth_rect.txt"
            ground_truth = list(csv.reader(open(ground_truth_file)))
            current_res = results[R_value][key]
            res[key].append(results[R_value][key])
    except:
        pass

for k, v in res.items():
    print(f"{k}", end = ' & ')
    for i, v1 in enumerate(v):
        print(f"${format(np.mean(v1), '.1f')}\pm{format(np.std(v1), '.1f')}$", end = ' & ')
        plt.plot(v1, label=f"R={R_values[i]}")
        plt.xlabel("Frames")
        plt.ylabel("Distance")
    print()
    plt.legend()
    plt.savefig(f"results/{k}-varying-R.pdf", format="pdf", bbox_inches='tight')
    plt.clf()

res = {}
print("Particle Filter point tracking with Q values 0.001, 0.01, 1, 10, 1000")
for j, Q_value in enumerate(Q_values):
    i = 2
    R_value = 1
    try:
        with open(f"results/particle_filter_point_R_{i}_Q_{j}_{args.object_detector}.pickle", 'rb') as f:
            results[Q_value] = pickle.load(f)
            if keys is None:
                keys = list(results[Q_value].keys())
        for key in keys:
            if key not in res:
                res[key] = []
            ground_truth_file = dataset + key + "/groundtruth_rect.txt"
            ground_truth = list(csv.reader(open(ground_truth_file)))
            current_res = results[Q_value][key]
            res[key].append(results[Q_value][key])
    except:
        pass

for k, v in res.items():
    print(f"{k}", end=' & ')
    for i, v1 in enumerate(v):
        print(f"${format(np.mean(v1), '.1f')}\pm{format(np.std(v1), '.1f')}$", end=' & ')
        plt.plot(v1, label=f"Q={Q_values[i]}")
        plt.xlabel("Frames")
        plt.ylabel("Distance")
    print()
    plt.legend()
    plt.savefig(f"results/{k}-varying-Q.pdf", format="pdf", bbox_inches='tight')
    plt.clf()
