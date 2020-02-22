import pickle
from glob import glob

import numpy as np

from association.ml_association import MLPFAssociation
from constants import Q_values, R_values
from density_extractor import DensityExtractor
from find_datasets import get_iou
from utils import (create_fig_ax, create_iterator, create_particle_filter,
                   extract_all_groud_truths, get_directories, load_object_detector, plot,
                   read_img, sort_images)
import os

os.makedirs("results", exist_ok=True)


def test_pf_R(args):
    results = {}
    fd = load_object_detector(args.object_detector)
    dataset = "data/TinyTLP/"
    all_directories = get_directories(dataset)

    n = 4 if args.point_estimate else 8
    Q_value = 0.01*np.eye(n)
    de = None

    for l, R_value in enumerate(R_values):
        for dir in all_directories:
            R_value = R_value * np.eye(4)
            results_dir = results[dir] = []
            iou = []
            ground_truth_file = dataset + dir + "/groundtruth_rect.txt"
            images_wildcard = dataset + dir + "/img/*.jpg"
            images_filelist = glob(images_wildcard)

            images_filelist = sort_images(images_filelist)

            if args.extract_density:
                de = DensityExtractor(images_filelist[0], args)
                de.create_grid()

            # Extract all ground truths
            ground_truth, gt_measurements = extract_all_groud_truths(
                ground_truth_file)

            # Create PF
            w, h = ground_truth[0][3:5]
            pf = create_particle_filter(
                ground_truth[0], Q_value, R_value, images_filelist[0], args)

            # Create images iterator
            t = create_iterator(images_filelist, args)

            # Create data association
            da = MLPFAssociation(states=pf.S, R=R_value, H=pf.H,
                                 threshold=args.outlier_detector_threshold)

            features = {}
            # Iterate of every image
            for i, im in enumerate(t):
                img = read_img(im)

                # Do prediction
                pf.predict()

                if i % args.resample_every == 0:

                    # Compute features
                    features[i] = np.array(fd.compute_features(img))

                    if len(features[i]) > 0:
                        # Do data association
                        if args.point_estimate:
                            features[i] = [
                                np.array([i[0] + w/2, i[1] + h/2]) for i in features[i]]
                        psi, outlier, c = da.associate(features[i])

                        pf.update(psi, outlier, c)
                    else:
                        pf.assign_predicted()

                else:
                    pf.assign_predicted()

                gt = list(map(int, ground_truth[i]))
                x = pf.get_x().T

                # Plot features
                if args.show_plots:
                    plot(args, x, de, img, gt, features[i])

                if args.extract_density:
                    de.estimate(x)
                    v = 0.0
                    if args.point_estimate:
                        v = np.linalg.norm(
                            [de.xtmean - gt[0] - gt[2]/2, de.ytmean - gt[1] - gt[3]/2], axis=0)
                        t.set_description(
                            f"pred_pos=({de.xtmean:3}, {de.ytmean:3}) | gt_pos=({round(gt[0] + w/2):3}, {round(gt[1] + h/2):3}) |"
                            " l2_dist={v:2.4}")
                    else:
                        v = np.linalg.norm([de.xtmean - gt[0] - (de.xbmean - de.xtmean)/2,
                                            de.ytmean -
                                            gt[1] - (de.ybmean - de.ytmean)/2,
                                            (de.xbmean - de.xtmean) - gt[2],
                                            (de.ybmean - de.ytmean) - gt[3]])
                    iou.append(get_iou([de.xtmean, de.ytmean, de.xbmean, de.ybmean],
                                       [gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                    results_dir.append(
                        [de.xtmean, de.ytmean, de.xbmean - de.xtmean, de.ybmean - de.ytmean])
                    with open(f"results/particles_filter_box_R_{l}_Q_2_{args.object_detector}.pickle", 'wb') as fp:
                        pickle.dump(results, fp)

def test_pf_Q(args):
    results = {}
    fd = load_object_detector(args.object_detector)
    dataset = "data/TinyTLP/"
    all_directories = get_directories(dataset)

    n = 4
    R_value = 750.0*np.eye(n)
    de = None

    for l, Q_value in enumerate(Q_values):
        for dir in all_directories:
            try:
                R_value = R_value * np.eye(4)
                results_dir = results[dir] = []
                iou = []
                results_dir = results[dir] = []
                ground_truth_file = dataset + dir + "/groundtruth_rect.txt"
                images_wildcard = dataset + dir + "/img/*.jpg"
                images_filelist = glob(images_wildcard)

                if args.extract_density:
                    de = DensityExtractor(images_filelist[0], args)
                    de.create_grid()

                # Extract all ground truths
                ground_truth, gt_measurements = extract_all_groud_truths(
                    ground_truth_file)

                # Create PF
                w, h = ground_truth[0][3:5]
                pf = create_particle_filter(
                    ground_truth[0], Q_value, R_value, images_filelist[0], args)

                # Create images iterator
                t = create_iterator(images_filelist, args)

                # Create data association
                da = MLPFAssociation(states=pf.S, R=R_value, H=pf.H,
                                     threshold=args.outlier_detector_threshold)

                if args.show_plots:
                    fig, ax = create_fig_ax()

                features = {}
                # Iterate of every image
                for i, im in enumerate(t):
                    img = read_img(im)

                    # Do prediction
                    pf.predict()

                    if i % args.resample_every == 0:

                        # Compute features
                        features[i] = np.array(fd.compute_features(img))

                        if len(features[i]) > 0:
                            # Do data association
                            if args.point_estimate:
                                features[i] = [
                                    np.array([i[0] + w/2, i[1] + h/2]) for i in features[i]]
                            psi, outlier, c = da.associate(features[i])

                            pf.update(psi, outlier, c)
                        else:
                            pf.assign_predicted()

                    else:
                        pf.assign_predicted()

                    gt = list(map(int, ground_truth[i]))
                    x = pf.get_x().T

                    # Plot features
                    if args.show_plots:
                        plot(args, ax, x, de, img, gt, features[i])

                    if args.extract_density:
                        de.estimate(x)
                        v = 0.0
                        if args.point_estimate:
                            v = np.linalg.norm(
                                [de.xtmean - gt[0] - gt[2]/2, de.ytmean - gt[1] - gt[3]/2], axis=0)
                            t.set_description(
                                f"pred_pos=({de.xtmean:3}, {de.ytmean:3}) | gt_pos=({round(gt[0] + w/2):3}, {round(gt[1] + h/2):3}) |"
                                " l2_dist={v:2.4}")
                        else:
                            v = np.linalg.norm([de.xtmean - gt[0] - (de.xbmean - de.xtmean)/2,
                                                de.ytmean -
                                                gt[1] - (de.ybmean - de.ytmean)/2,
                                                (de.xbmean - de.xtmean) - gt[2],
                                                (de.ybmean - de.ytmean) - gt[3]])
                        iou.append(get_iou([de.xtmean, de.ytmean, de.xbmean, de.ybmean],
                                           [gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                        results_dir.append(
                            [de.xtmean, de.ytmean, de.xbmean - de.xtmean, de.ybmean - de.ytmean])
                        with open(f"results/particles_filter_box_R_2_Q_{l}_{args.object_detector}.pickle", 'wb') as fp:
                            pickle.dump(results, fp)
            except np.linalg.LinAlgError:
                print("jumped")
