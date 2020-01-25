import csv
import matplotlib
matplotlib.use("Qt5Cairo")
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from tqdm import tqdm

from association.ml_association import MLAssociation as DataAssociation
from kalman_filter import KalmanFilter
from yolo.yolo_object_detection import YOLOObjectDetection as FeaturesDetector


def main():
    diff = 0

    # Get images list from dataset
    dataset = "data/jogging"
    ground_truth_file = dataset + "/groundtruth_rect.1.txt"
    images_wildcard = dataset + "/img/*.jpg"
    images_filelist = glob(images_wildcard)

    # Sort them in ascending order
    images_filelist = sorted(images_filelist, key=lambda xx: int(
        xx.split('/')[-1].split('.')[0]))

    # Extract all ground truths
    ground_truth = list(csv.reader(open(ground_truth_file)))

    initial_position = ground_truth[0]
    x, y, w, h = initial_position
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    # img = plt.imread(images_filelist[0])
    # plt.imshow(img)
    # r = mpatches.Rectangle((x, y), w, h, linewidth=1, facecolor="none", edgecolor="red")
    # ax = plt.gca()
    # ax.add_patch(r)
    # plt.show()

    Q = np.diag([2, 2, 2, 2])
    # Initialize KF (x = [x, vx, y, vy])
    kf = KalmanFilter(x=np.array([[x+w/2], [1], [y+y/2], [1]]), Q=Q)

    # Initialize features detector
    fd = FeaturesDetector()

    # Find Q using EM
    # em_obj = EM()
    # data = np.array(measurements)
    # likelihoods = em_obj.find_Q(data)
    # print(em_obj.kf.Q)
    # kf.Q = em_obj.kf.Q
    # plt.plot(likelihoods)
    # plt.show()

    # Iterate of every image
    features = {}
    t = tqdm(images_filelist[1:], desc="Processing")

    da = DataAssociation(R=kf.R, H=kf.H, threshold=50)

    plt.ion()
    for i, im in enumerate(t):
        img = plt.imread(images_filelist[i])
        # Compute features
        features[i] = np.array(fd.compute_features(im))

        # Do prediction
        mu_bar, Sigma_bar = kf.predict()

        # Do data association
        da.update_prediction(mu_bar, Sigma_bar)
        m = da.associate(im, features[i])

        kf.update(m)

        gt = list(map(int, ground_truth[i]))

        plt.gca()
        plt.cla()
        plt.imshow(img)
        plt.plot(kf.x[0][0], kf.x[2][0], marker='o', color='blue')
        plt.pause(0.0001)

        print(
            f"Predicted position: {kf.x[0][0], kf.x[2][0]}, Ground truth position: {gt[0] + w/2, gt[1] + h/2}")
        diff += np.linalg.norm([kf.x[0][0] - gt[0] - w/2, kf.x[2][0] - gt[1] - h/2], axis=0)

    print(diff / len(images_filelist))


if __name__ == '__main__':
    main()
