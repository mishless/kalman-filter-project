import csv

import matplotlib

from expectation_maximisation import EM

matplotlib.use("Qt5Cairo")
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from association.ml_association import MLPFAssociation as DataAssociation
from particle_filter import ParticleFilter
from yolo.yolo_object_detection import YOLOObjectDetection as FeaturesDetector

import scipy.stats as st

def main():
    diff = 0

    # Get images list from dataset
    dataset = "data/Car4"
    ground_truth_file = dataset + "/groundtruth_rect.txt"
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
    x = int(float(x) + float(w)/2)
    y = int(float(y) + float(h)/2)
    w = int(w)
    h = int(h)

    resample_every: int = 1  # n: resample once every n frames
    plot_particles: bool = False  # If true, draw green dots on the particles positions
    plot_ground_truth: bool = True  # If true, plot a yellow bounding box on the tracked object
    plot_detector_output: bool = False  # If true, plot a blue bounding box on the output of the detector
    use_em_to_find_q: bool = False  # If true, use EM to estimate the Q matrix (experimental)
    global_tracking: bool = True  # If true, use the first ground truth datum to initialize the estimator on the target as a prior.
    extract_density: bool = True  # If true, extract density of the results. True required for plot_density and plot_detected_mean.
    plot_density: bool = True  # to be used with extract density
    plot_detected_mean: bool = True  # to be used with extract density

    R = 750 * np.eye(2)
    # Initialize KF (x = [x, vx, y, vy])
    M = 1000
    pf = ParticleFilter(num_particles=M, R=R, img_shape=img.shape, resample_mode="multinomial", initial_state=(x, y))

    if not global_tracking:
        pf.S[:, :2] = np.random.randn()

    # Initialize features detector
    fd = FeaturesDetector()

    # Find Q using EM
    if use_em_to_find_q:
        em_obj = EM()
        data = np.array(ground_truth)[:, :2].astype(int)
        pf.Q = em_obj.kf.Q.astype(int)*0.01
    else:
        pf.Q = 0.05 * np.eye(4)

    # Iterate of every image
    features = {}
    t = tqdm(images_filelist[1:], desc="Processing")

    da = DataAssociation(states=pf.S, R=pf.R, H=pf.H, threshold=1e-5)

    fig, ax = plt.subplots(1, 1)
    plt.ion()

    # Define the borders
    xmin = 0
    xmax = img.shape[1]
    ymin = 0
    ymax = img.shape[0]
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    for i, im in enumerate(t):
        img = plt.imread(images_filelist[i])
        plt.gca()
        plt.cla()
        plt.imshow(img)

        # Do prediction
        pf.predict()

        if i % resample_every == 0:

            # Compute features
            features[i] = np.array(fd.compute_features(im))

            # Plot features
            if plot_detector_output:
                for f in features[i]:
                    r = mpatches.Rectangle((f[0] - w/2, f[1] - h/2), w, h, linewidth=1, linestyle="solid",
                                           edgecolor="r", facecolor=None, fill=None)
                    ax.add_patch(r)

            # Do data association
            psi, outlier, c = da.associate(features[i])

            pf.update(psi, outlier, c)

        else:
            pf.assign_predicted()

        gt = list(map(int, ground_truth[i]))
        if plot_ground_truth:
            r = mpatches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, linestyle="solid",
                                   edgecolor="y", facecolor=None, fill=None)
            ax.add_patch(r)
        x = pf.get_x().T
        plt.gca().autoscale(False)
        if extract_density:
            kernel = st.gaussian_kde(x)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)
            imean = np.argmax(f)
            xmean = int(round(xx[imean // 100, imean % 100]))
            ymean = int(round(yy[imean // 100, imean % 100]))
        if plot_particles:
            plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
        if plot_density and extract_density:
            f[f < 1e-5] = np.nan
            cfset = ax.contourf(xx, yy, f, cmap='coolwarm', alpha=0.5)
            cset = ax.contour(xx, yy, f, colors='k', alpha=0.5)
        if plot_detected_mean and extract_density:
            r = mpatches.Rectangle((xmean - w/2, ymean - h/2), gt[2], gt[3], linewidth=1, linestyle="solid",
                                   edgecolor="b", facecolor=None, fill=None)
            ax.add_patch(r)
            v = np.linalg.norm([xmean - gt[0] - w/2, ymean - gt[1] - h/2], axis=0)
            t.set_description(f"pred_pos=({xmean:3}, {ymean:3}) | gt_pos=({round(gt[0] + w/2):3}, {round(gt[1] + h/2):3}) | l2_dist={v:2.4}")
            diff += v
        plt.pause(0.1)

    print("Mean euclidean distance:", diff / len(images_filelist))


if __name__ == '__main__':
    main()
