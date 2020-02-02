import csv

import matplotlib

matplotlib.use("Qt5Cairo")
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
from tqdm import tqdm

from association.ml_association import MLPFAssociation as DataAssociation
from particle_filter import ParticleFilter
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
    img = plt.imread(images_filelist[0])

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

    Q = 10 * np.eye(4)
    R = 100 * np.eye(2)
    # Initialize KF (x = [x, vx, y, vy])
    M = 10000
    pf = ParticleFilter(num_particles=M, Q=Q, R=R, img_shape=img.shape, resample_mode="multinomial")

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

    da = DataAssociation(states=pf.S, R=pf.R, H=pf.H, threshold=1e-7)

    # Plot initialized states
    # x = pf.get_x().T
    # plt.gca()
    # plt.cla()
    # plt.imshow(img)
    # plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
    # plt.pause(0.1)

    plt.ion()
    for i, im in enumerate(t):
        img = plt.imread(images_filelist[i])
        # Compute features
        features[i] = np.array(fd.compute_features(im))

        # Do prediction
        pf.predict()

        # Plot predicted
        # plt.gca()
        # plt.cla()
        # plt.imshow(img)
        # x = pf.S_bar[:, [0, 2]].T
        # plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
        # plt.xlim(0, img.shape[1])
        # plt.ylim(0, img.shape[0])
        # plt.pause(0.1)

        # Do data association
        psi, outlier, c = da.associate(features[i])

        pf.update(psi, outlier, c)

        gt = list(map(int, ground_truth[i]))

        x = pf.get_x().T

        plt.gca()
        plt.cla()
        plt.imshow(img)
        plt.gca().autoscale(False)
        plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
        plt.pause(0.1)

        # print(
        #     f"Predicted position: {pf.x[0][0], pf.x[2][0]}, Ground truth position: {gt[0] + w/2, gt[1] + h/2}")
        # diff += np.linalg.norm([pf.x[0][0] - gt[0] - w/2, pf.x[2][0] - gt[1] - h/2], axis=0)

    print(diff / len(images_filelist))


if __name__ == '__main__':
    main()
