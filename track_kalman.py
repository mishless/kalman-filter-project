import csv
#matplotlib.use("Qt5Cairo")
from glob import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from association.ml_association import MLKalmanAssociation as DataAssociation
from kalman_filter import KalmanFilter
from yolo.yolo_object_detection import YOLOObjectDetection
from ssd.ssd import SSDObjectDetection
from scipy.stats import multivariate_normal
import os
import pickle

def main():
    R_values = [0.001, 0.1, 1, 10, 1000]
    Q_values = [0.001, 0.1, 1, 10, 1000]
    parser = argparse.ArgumentParser(
        description="Run SSD on input folder and show result in popup window")
    parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                        help="Specify which object detector network should be used")
    parser.add_argument("test", choices=['Q', 'R'],
                        help="Which noise matrix should be tested")
    args = parser.parse_args()
    if args.object_detector == 'ssd':
        fd = SSDObjectDetection(frozen, pbtxt)
    elif args.object_detector == 'yolo_full':
        fd = YOLOObjectDetection('full')
    elif args.object_detector == 'yolo_tiny':
        fd = YOLOObjectDetection('tiny')

    # Config
    should_plot = False

    # Get images list from dataset
    dataset = "data/TinyTLP/"

    for path, directories, files in os.walk(dataset):
        all_directories = directories
        break
    results = {}
    if args.test == 'R':
        k = 2
        Q_value = 1
        for l, R_value in enumerate(R_values):
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

                initial_position = ground_truth[0]
                frame_id, x, y, w, h, is_lost = initial_position
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                kf = KalmanFilter(x=np.array([[x + w / 2], [1], [y + h / 2], [1]]), Q=Q_value, R=R_value)


                # Iterate of every image
                features = {}
                t = tqdm(images_filelist[1:], desc="Processing")

                da = DataAssociation(R=kf.R, H=kf.H, threshold=0.1)

                plt.ion()
                for i, im in enumerate(t):
                    img = plt.imread(images_filelist[i])
                    height, width, _ = img.shape
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
                        x, y = np.mgrid[0:width, 0:height]
                        pos = np.empty(x.shape + (2,))
                        pos[:, :, 0] = x
                        pos[:, :, 1] = y
                        rv = multivariate_normal([kf.x[0][0], kf.x[2][0]], [[kf.cov[0][0], kf.cov[0][2]], [kf.cov[2][0], kf.cov[2][2]]])
                        f = rv.pdf(pos)
                        f[f < 1e-5] = np.nan

                        plt.gca()
                        plt.cla()
                        plt.imshow(img)
                        plt.contourf(x, y, f, cmap='coolwarm', alpha=0.5)
                        if m is not None:
                            plt.plot(m[0], m[1], marker='o', color='yellow')
                        plt.plot(gt[1] + w/2, gt[2] + h/2, marker='o', color='red')
                        plt.pause(0.0001)

                        print(f"Diff: {np.linalg.norm([kf_x[0] - gt[1] - w / 2, kf_x[1] - gt[2] - h / 2])} Predicted position: {kf_x[0], kf_x[1]}, Ground truth position: {gt[1] + w / 2, gt[2] + h / 2}")
                    results[dir].append(np.linalg.norm([kf_x[0] - gt[1] - w / 2, kf_x[1] - gt[2] - h / 2]))

                print(f"Dataset: {dir} mean distance: {np.mean(results[dir])}, std: {np.std(results[dir])}")
            with open(f"results/kalman_filter_point_R_{l}_Q_{k}_{args.object_detector}.pickle", 'wb') as fp:
                pickle.dump(results, fp)
    elif args.test == 'Q':
        l = 2
        R_value = 1
        for k, Q_value in enumerate(Q_values):
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

                initial_position = ground_truth[0]
                frame_id, x, y, w, h, is_lost = initial_position
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                kf = KalmanFilter(x=np.array([[x + w / 2], [1], [y + h / 2], [1]]), Q=Q_value, R=R_value)


                # Iterate of every image
                features = {}
                t = tqdm(images_filelist[1:], desc="Processing")

                da = DataAssociation(R=kf.R, H=kf.H, threshold=0.1)

                plt.ion()
                for i, im in enumerate(t):
                    img = plt.imread(images_filelist[i])
                    height, width, _ = img.shape
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
                        x, y = np.mgrid[0:width, 0:height]
                        pos = np.empty(x.shape + (2,))
                        pos[:, :, 0] = x
                        pos[:, :, 1] = y
                        rv = multivariate_normal([kf.x[0][0], kf.x[2][0]], [[kf.cov[0][0], kf.cov[0][2]], [kf.cov[2][0], kf.cov[2][2]]])
                        f = rv.pdf(pos)
                        f[f < 1e-5] = np.nan

                        plt.gca()
                        plt.cla()
                        plt.imshow(img)
                        plt.contourf(x, y, f, cmap='coolwarm', alpha=0.5)
                        if m is not None:
                            plt.plot(m[0], m[1], marker='o', color='yellow')
                        plt.plot(gt[1] + w/2, gt[2] + h/2, marker='o', color='red')
                        plt.pause(0.0001)

                        print(f"Diff: {np.linalg.norm([kf_x[0] - gt[1] - w / 2, kf_x[1] - gt[2] - h / 2])} Predicted position: {kf_x[0], kf_x[1]}, Ground truth position: {gt[1] + w / 2, gt[2] + h / 2}")
                    results[dir].append(np.linalg.norm([kf_x[0] - gt[1] - w / 2, kf_x[1] - gt[2] - h / 2]))

                print(f"Dataset: {dir} mean distance: {np.mean(results[dir])}, std: {np.std(results[dir])}")
            with open(f"results/kalman_filter_point_R_{l}_Q_{k}_{args.object_detector}.pickle", 'wb') as fp:
                pickle.dump(results, fp)


if __name__ == '__main__':
    main()
