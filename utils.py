import argparse
import csv
import os
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

from constants import frozen, pbtxt
from particle_filter import ParticleFilter
from ssd.ssd import SSDObjectDetection
from yolo.yolo_object_detection import YOLOObjectDetection

from io import BytesIO


def rectify(array,
            bounds=None):
    if bounds is None:
        bounds = {0: (0, 100), 2: (0, 100)}
    for c, (b0, b1) in bounds.items():
        col = array[:, c]
        col[col < b0] = b0
        col[col > b1] = b1


def parse_arguments_kf():
    parser = argparse.ArgumentParser(
        description="Run SSD on input folder and show result in popup window")
    parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                        help="Specify which object detector network should be used")
    parser.add_argument("test", choices=['Q', 'R'],
                        help="Which noise matrix should be tested")
    parser.add_argument("should_plot", actions="store_true",
                        help="Whether or not to plot the boxes")
    args = parser.parse_args()
    return args


def parse_arguments_pf():
    parser = argparse.ArgumentParser(
        description="Run PF tests")
    parser.add_argument("object_detector", choices=['ssd', 'yolo_full', 'yolo_tiny'],
                        help="Specify which object detector network should be used")
    parser.add_argument("--should_plot", action="store_true",
                        help="Whether or not to plot the boxes")
    parser.add_argument("--resample_every", type=int, default=1,
                        help="n: resample once every n frames")
    parser.add_argument("--plot_particles", help="Draw green dots on the particles positions", action="store_true")
    parser.add_argument("--plot_ground_truth", help="Plot a yellow bounding box on the tracked object",
                        action="store_true")
    parser.add_argument("--plot_detector_output",
                        help="Plot a blue bounding box on the output of the detector", action="store_true")
    parser.add_argument("--global_tracking",
                        help="Use the first ground truth datum to "
                             "initialize the estimator on the target as a prior.",
                        action="store_true")
    parser.add_argument("--extract_density", help="Extract density of the results. Required for "
                                                  "plot_density and plot_detected_mean.", action="store_true")
    parser.add_argument(
        "--plot_density", help="To be used with extract density", action="store_true")
    parser.add_argument("--plot_detected_mean",
                        help="To be used with extract density", action="store_true")
    parser.add_argument("--resample_mode",
                        help="Mode of resampling, either 'systematic' or 'multinomial'", default="systematic", type=str)
    parser.add_argument("--point_estimate",
                        help="Estimate just the mean point of the target and not the bounding box",
                        action="store_true")
    parser.add_argument("--object_detector_confidence", type=float, default=0.1,
                        help="Level of confidence of the YOLO object detector")
    parser.add_argument("--num_particles", type=int, default=10000,
                        help="Number of particles")
    parser.add_argument("--outlier_detector_threshold", type=float, default=1e-9,
                        help="Theshold mahalonobis distance for outliers detection")
    parser.add_argument("--show_plots", action="store_true", help="Show plots")

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


def create_particle_filter(initial_position, Q, R, initial_img_filename, args):
    initial_img = plt.imread(initial_img_filename)
    initial_position = initial_position
    frame_id, x, y, w, h, is_lost = initial_position
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    initial_state = None
    if args.point_estimate:
        initial_state = (x+w/2, y+h/2)
    else:
        if not args.global_tracking:
            initial_state = (x, y, x+w, y+h)

    # Initialize PF
    pf = ParticleFilter(num_particles=args.num_particles, R=R, img_shape=initial_img.shape,
                        resample_mode=args.resample_mode, initial_state=initial_state,
                        point_estimate=args.point_estimate)
    return pf


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


def create_fig_ax():
    return plt.subplots(1)


def read_img(img_file):
    return plt.imread(img_file)


def create_iterator(images_filelist, args):
    if args.global_tracking:
        start = 0
    else:
        start = 1
    t = tqdm(images_filelist[start:], desc="Processing")
    return t


def plot_detector_output(ax, detector_output):
    for f in detector_output:
        r = patches.Rectangle((f[0], f[1]), f[2]-f[0], f[3]-f[1], linewidth=1, linestyle="solid",
                              edgecolor="r", facecolor=None, fill=None)
        ax.add_patch(r)


def plot_ground_truth(ax, gt):
    r = patches.Rectangle((gt[0], gt[1]), gt[2], gt[3], linewidth=1, linestyle="solid", edgecolor="y",
                          facecolor=None, fill=None)
    ax.add_patch(r)


def plot(args, x=None, de=None, img=None, gt=None, detector_output=None):
    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.autoscale(False)

    if args.plot_detector_output:
        plot_detector_output(ax, detector_output)

    if args.plot_ground_truth:
        plot_ground_truth(ax, gt)

    if args.plot_particles:
        if args.point_estimate:
            plt.plot(x[0].astype(int), x[1].astype(int), 'g.')
        else:
            for xi in sample(x.T.tolist(), 10):
                r = patches.Rectangle((int(xi[0]), int(xi[1])), int(abs(xi[2]-xi[0])), int(abs(xi[3]-xi[1])),
                                      linewidth=1, linestyle="solid", edgecolor="y", facecolor=None, fill=None)
                ax.add_patch(r)
    if args.extract_density:
        e = de.estimate(x)
        if args.point_estimate:
            f, xtmean, ytmean = e
            xbmean = ybmean = 0
        else:
            f1, f2, (xtmean, ytmean), (xbmean, ybmean) = e

        if args.plot_density and args.point_estimate:
                f[f < 1e-5] = np.nan
                cfset = ax.contourf(de.xx, de.yy, f, cmap='coolwarm', alpha=0.5)
                cset = ax.contour(de.xx, de.yy, f, colors='k', alpha=0.5)
        if args.plot_detected_mean:
            if args.point_estimate:
                w = gt[2]
                h = gt[3]
                r = patches.Rectangle((xtmean - w/2, ytmean - h/2), w, h, linewidth=1, linestyle="solid",
                                      edgecolor="b", facecolor=None, fill=None)
            else:
                w = xbmean - xtmean
                h = ybmean - ytmean
                r = patches.Rectangle((xtmean, ytmean), w, h, linewidth=1, linestyle="solid",
                                      edgecolor="b", facecolor=None, fill=None)
            ax.add_patch(r)

    ax.axis("off")

    if args.show_plots:
        plt.show()
        return None
    else:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        img = Image.open(buf)
        return img


def sort_images(unsorted_filelist):
    def extract_id(filename):
        return int(filename.split(".")[0].split("/")[-1])
    return sorted(unsorted_filelist, key=extract_id)