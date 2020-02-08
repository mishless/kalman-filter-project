import csv

import matplotlib

#matplotlib.use("Qt5Cairo")
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from association.ml_association import MLKalmanAssociation as DataAssociation
from kalman_filter import KalmanFilter
from yolo.yolo_object_detection import YOLOObjectDetection as FeaturesDetector
from scipy.stats import multivariate_normal
from expectation_maximisation import EM

def main():
    diff = 0


    # Get images list from dataset
    dataset = "data/car"
    ground_truth_file = dataset + "/groundtruth_rect.txt"
    images_wildcard = dataset + "/img/*.jpg"
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
    x, y, w, h = initial_position
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    #Q = np.diag([2, 2, 2, 2])
    Q = 0.05 * np.eye(4)
    R = 750 * np.eye(2)
    # Initialize KF (x = [x, vx, y, vy])
    kf = KalmanFilter(x=np.array([[x + w / 2], [1], [y + y / 2], [1]]), Q=Q, R=R)

    # Initialize features detector
    fd = FeaturesDetector()

    # Find Q using EM
    em_obj = EM()
    data = np.array(gt_measurements)
    likelihoods = em_obj.find_Q(data)
    print(em_obj.kf.Q)
    kf.Q = em_obj.kf.Q
    plt.plot(likelihoods)
    plt.show()

    # Iterate of every image
    features = {}
    t = tqdm(images_filelist[1:], desc="Processing")

    da = DataAssociation(R=kf.R, H=kf.H, threshold=50)

    plt.ion()
    for i, im in enumerate(t):
        img = plt.imread(images_filelist[i])
        height, width = img.shape
        # Compute features
        features[i] = np.array(fd.compute_features(im))

        # Do prediction
        mu_bar, Sigma_bar = kf.predict()

        # Do data association
        da.update_prediction(mu_bar, Sigma_bar)
        m = da.associate(features[i])

        kf.update(m)

        gt = list(map(int, ground_truth[i]))

        kf_x = kf.get_x()

        x, y = np.mgrid[0:width, 0:height]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = multivariate_normal([kf.x[0][0], kf.x[2][0]], [[kf.cov[0][0], kf.cov[0][2]], [kf.cov[2][0], kf.cov[2][2]]])
        f = rv.pdf(pos)
        f[f < 1e-5] = np.nan
        #print(kf.x[2][0], kf.x[0][0])
        #print([[kf.cov[0][0], kf.cov[0][2]], [kf.cov[2][0], kf.cov[2][2]]])
        #print(rv.pdf(pos))
        plt.gca()
        plt.cla()
        plt.imshow(img)
        plt.contourf(x, y, f, cmap='coolwarm', alpha=0.5)
        #plt.plot(kf_x[0], kf_x[1], marker='o', color='blue')
        plt.pause(0.0001)
        print(
            f"Predicted position: {kf_x[0], kf_x[1]}, Ground truth position: {gt[0] + w / 2, gt[1] + h / 2}")
        diff += np.linalg.norm([kf_x[0] - gt[0] - w / 2, kf_x[1] - gt[1] - h / 2], axis=0)

    print(diff / len(images_filelist))


if __name__ == '__main__':
    main()
