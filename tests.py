from glob import glob
import numpy as np
from tqdm import tqdm
from association.ml_associations_box import MLKalmanAssociation as DataAssociation
from find_datasets import get_iou
import pickle
from constants import R_values, Q_values
from utils import load_object_detector, get_directories, extract_all_groud_truths, create_box_kalman_filter
from utils import plot_box_kf_result, create_fig_ax, read_img


def test_R(dataset, all_directories, args):
    results = {}
    k = 2
    Q_value = 1
    # Get images list from dataset
    dataset = "data/TinyTLP/"

    all_directories = get_directories(dataset)
    fd = load_object_detector(args.object_detector)
    for l, R_value in enumerate(R_values):
        for dir in all_directories:
            iou = []
            results_dir = results[dir] = []
            ground_truth_file = dataset + dir + "/groundtruth_rect.txt"
            images_wildcard = dataset + dir + "/img/*.jpg"
            images_filelist = glob(images_wildcard)

            # Extract all ground truths
            ground_truth, gt_measurements = extract_all_groud_truths(ground_truth_file)

            # Create KF
            kf = create_box_kalman_filter(ground_truth[0], Q_value, R_value)

            # Iterate of every image
            t = tqdm(images_filelist[1:], desc="Processing")
            da = DataAssociation(R=kf.R, H=kf.H, threshold=1)

            # Create plot if should_plot is true
            if args.should_plot:
                fig, ax = create_fig_ax()

            features = {}
            for i, im in enumerate(t):
                img = read_img(images_filelist[i])
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

                if args.should_plot:
                    plot_box_kf_result(ax, i, img, m, kf_x, ground_truth)
                iou.append(get_iou([kf_x[0], kf_x[1], kf_x[2], kf_x[3]], [
                           gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                results_dir.append(
                    [kf_x[0], kf_x[1], kf_x[2] - kf_x[0], kf_x[3] - kf_x[1]])
            print(f"Dataset: {dir}, IoU: {np.mean(iou), np.std(iou)}")
        with open(f"results/kalman_filter_box_R_{l}_Q_{k}_{args.object_detector}.pickle", 'wb') as fp:
            pickle.dump(results, fp)


def test_Q(dataset, all_directories, args):
    results = {}
    l = 2
    R_value = 1
    # Get images list from dataset
    dataset = "data/TinyTLP/"

    all_directories = get_directories(dataset)
    fd = load_object_detector(args.object_detector)
    for k, Q_value in enumerate(Q_values):
        for dir in all_directories:
            iou = []
            results_dir = results[dir] = []
            ground_truth_file = dataset + dir + "/groundtruth_rect.txt"
            images_wildcard = dataset + dir + "/img/*.jpg"
            images_filelist = glob(images_wildcard)

            # Extract all ground truths
            ground_truth, gt_measurements = extract_all_groud_truths(ground_truth_file)

            # Create KF
            kf = create_box_kalman_filter(ground_truth[0], Q_value, R_value)

            # Iterate of every image
            t = tqdm(images_filelist[1:], desc="Processing")
            da = DataAssociation(R=kf.R, H=kf.H, threshold=1)

            # Create plot if should_plot is true
            if args.should_plot:
                fig, ax = create_fig_ax()

            features = {}
            for i, im in enumerate(t):
                img = read_img(images_filelist[i])
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

                if args.should_plot:
                    plot_box_kf_result(ax, i, img, m, kf_x, ground_truth)
                iou.append(get_iou([kf_x[0], kf_x[1], kf_x[2], kf_x[3]], [
                           gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                results_dir.append(
                    [kf_x[0], kf_x[1], kf_x[2] - kf_x[0], kf_x[3] - kf_x[1]])
            print(f"Dataset: {dir}, IoU: {np.mean(iou), np.std(iou)}")
        with open(f"results/kalman_filter_box_R_{l}_Q_{k}_{args.object_detector}.pickle", 'wb') as fp:
            pickle.dump(results, fp)
