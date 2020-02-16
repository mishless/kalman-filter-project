import scipy.stats as st
from yolo.yolo_object_detection import YOLOObjectDetection as FeaturesDetector
from particle_filter import ParticleFilter
from association.ml_association import MLPFAssociation as DataAssociation
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from glob import glob
import csv
import matplotlib
from expectation_maximisation import EM

matplotlib.use("Qt5Cairo")

# Parameters
resample_every: int = 1  # n: resample once every n frames
plot_particles: bool = True  # If true, draw green dots on the particles positions
# If true, plot a yellow bounding box on the tracked object
plot_ground_truth: bool = False
# If true, plot a blue bounding box on the output of the detector
plot_detector_output: bool = False
# If true, use EM to estimate the Q matrix (experimental)
use_em_to_find_q: bool = False
# If true, use the first ground truth datum to initialize the estimator on the target as a prior.
global_tracking: bool = False
# If true, extract density of the results. True required for plot_density and plot_detected_mean.
extract_density: bool = False
plot_density: bool = False  # to be used with extract density
plot_detected_mean: bool = False  # to be used with extract density
resample_mode = "systematic"

point_estimate = False


if point_estimate:
    outlier_detector_threshold = 1e-6
    num_particles = 1000
    R = 750*np.eye(2)
    Q = 0.01*np.eye(4)
else:
    outlier_detector_threshold = 1e-9
    num_particles = 10000
    R = 1500*np.eye(4)
    Q = 0.01*np.eye(8)
yolo_confidence = 0.1


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
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    # Initialize PF
    initial_state = None
    if point_estimate:
        initial_state = (x+w/2, y+h/2)
    else:
        if not global_tracking:
            initial_state = (x, y, x+w, y+h)
    pf = ParticleFilter(num_particles=num_particles, R=R, img_shape=img.shape, resample_mode=resample_mode,
                        initial_state=initial_state, point_estimate=point_estimate)

    # Initialize features detector
    fd = FeaturesDetector(confidence=yolo_confidence)

    # Find Q using EM
    if use_em_to_find_q:
        em_obj = EM()
        data = np.array(ground_truth)[:, :4].astype(int)
        data[:, 2:] += data[:, :2]
        pf.Q = em_obj.find_Q(data)
    else:
        pf.Q = Q

    # Iterate of every image
    features = {}
    if global_tracking:
        start = 0
    else:
        start = 1
    t = tqdm(images_filelist[start:], desc="Processing")

    da = DataAssociation(states=pf.S, R=pf.R, H=pf.H,
                         threshold=outlier_detector_threshold)

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
        img = plt.imread(im)
        plt.gca()
        plt.cla()
        plt.imshow(img)

        # Do prediction
        pf.predict()

        if i % resample_every == 0:

            # Compute features
            features[i] = np.array(fd.compute_features(im))

            # Filter out size <=0 features
            features[i] = list(filter(lambda x: x[2]-x[0] >
                                      0 and x[3]-x[1] > 0, features[i]))

            # Plot features
            if plot_detector_output:
                for f in features[i]:
                    r = mpatches.Rectangle((f[0], f[1]), f[2]-f[0], f[3]-f[1], linewidth=1, linestyle="solid",
                                           edgecolor="r", facecolor=None, fill=None)
                    ax.add_patch(r)

            # Do data association
            if point_estimate:
                features[i] = [np.array([i[0] + w/2, i[1] + h/2]) for i in features[i]]
            psi, outlier, c = da.associate(features[i])

            pf.update(psi, outlier, c)

        else:
            pf.assign_predicted()

        gt = list(map(int, ground_truth[i]))
        if plot_ground_truth:
            r = mpatches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, linestyle="solid", edgecolor="y",
                                   facecolor=None, fill=None)
            ax.add_patch(r)
        x = pf.get_x().T
        plt.gca().autoscale(False)
        if extract_density:
            if point_estimate:
                data = x
            else:
                data = np.vstack((x[0] + x[2], x[1] + x[3])) // 2
            kernel = st.gaussian_kde(data)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)
            imean = np.argmax(f)
            xmean = int(round(xx[imean // 100, imean % 100]))
            ymean = int(round(yy[imean // 100, imean % 100]))
        if plot_particles:
            if point_estimate:
                plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
            else:
                for xi in sample(x.T.tolist(), 10):
                    r = mpatches.Rectangle((int(xi[0]), int(xi[1])), int(abs(xi[2]-xi[0])), int(abs(xi[3]-xi[1])),
                                           linewidth=1, linestyle="solid", edgecolor="y", facecolor=None, fill=None)
                    ax.add_patch(r)
        if plot_density and extract_density:
            f[f < 1e-5] = np.nan
            cfset = ax.contourf(xx, yy, f, cmap='coolwarm', alpha=0.5)
            cset = ax.contour(xx, yy, f, colors='k', alpha=0.5)
        if plot_detected_mean and extract_density:
            r = mpatches.Rectangle((xmean - w/2, ymean - h/2), gt[2], gt[3], linewidth=1, linestyle="solid",
                                   edgecolor="b", facecolor=None, fill=None)
            ax.add_patch(r)
            v = np.linalg.norm(
                [xmean - gt[0] - w/2, ymean - gt[1] - h/2], axis=0)
            t.set_description(
                f"pred_pos=({xmean:3}, {ymean:3}) | gt_pos=({round(gt[0] + w/2):3}, {round(gt[1] + h/2):3}) |"
                " l2_dist={v:2.4}")
            diff += v
        plt.pause(0.1)

    print("Mean euclidean distance:", diff / len(images_filelist))


if __name__ == '__main__':
    main()
