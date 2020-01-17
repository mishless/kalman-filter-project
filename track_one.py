import csv
from tqdm import tqdm
from glob import glob
from kalman_filter import KalmanFilter
from expectation_maximisation import EM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolo.yolo_object_detection import YOLOObjectDetection as FeaturesDetector
from association.ml_association import MLAssociation as DataAssociation


def main():
    diff = 0

    # Get images list from dataset
    dataset = "data/jogging"
    ground_truth_file = dataset + "/groundtruth_rect.1.txt"
    images_wildcard = dataset + "/img/*.jpg"
    images_filelist = glob(images_wildcard)

    # Sort them in ascending order
    images_filelist = sorted(images_filelist, key=lambda x: int(
        x.split('/')[-1].split('.')[0]))

    # Extract all ground truths
    ground_truth = list(csv.reader(open(ground_truth_file)))

    initial_position = ground_truth[0]
    x, y, w, h = initial_position
    x = int(x)
    y = int(y)

    # Initialize KF (x = [x, vx, y, vy])
    kf = KalmanFilter(x=np.array([[x], [20], [y], [20]]))

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

    da = DataAssociation(R=kf.R, H=kf.H, threshold=5)

    for i, im in enumerate(t):
        # Compute features
        features[i] = np.array(fd.compute_features(im))

        # Do prediction
        mu_bar, Sigma_bar = kf.predict()

        # Do data association
        da.update_prediction(mu_bar, Sigma_bar)
        m = da.associate(im, features[i])

        kf.update(m)

        gt = list(map(int, ground_truth[i]))

        print(
            f"Predited position: {kf.x[0][0], kf.x[2][0]}, Ground truth position: {gt[0], gt[1]}")
        diff += np.linalg.norm([kf.x[0][0] - gt[0], kf.x[2][0] - gt[1]], axis=0)

    print(diff/len(images_filelist))


if __name__ == '__main__':
    main()
