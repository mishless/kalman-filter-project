import pickle
from glob import glob

import numpy as np

from association.ml_association import MLPFAssociation
from constants import Q_values, R_values
from density_extractor import DensityExtractor
from find_datasets import get_iou
from utils import (create_iterator, create_particle_filter,
                   extract_all_groud_truths, get_directories, load_object_detector, plot,
                   read_img, sort_images)
import os

os.makedirs("results", exist_ok=True)


def test_pf(args):
    fd = load_object_detector(args.object_detector)
    dataset = "data/TinyTLP/"
    all_directories = get_directories(dataset)

    n = 4 if args.point_estimate else 8
    de = None

    for q_ind, q_value in enumerate(Q_values):
        q_value = 0.01*np.eye(n)
        for r_ind, r_value in enumerate(R_values):
            results = {}
            try:
                r_value = r_value * np.eye(4)
                for directory in all_directories:
                    results_dir = results[directory] = []
                    plotted = results[f"{directory}_plots"] = []
                    iou = results[f"{directory}_iou"] = []
                    errors = results[f"{directory}_errors"] = []
                    ground_truth_file = dataset + directory + "/groundtruth_rect.txt"
                    images_wildcard = dataset + directory + "/img/*.jpg"
                    images_filelist = glob(images_wildcard)

                    images_filelist = sort_images(images_filelist)

                    if args.extract_density:
                        de = DensityExtractor(images_filelist[0], args)
                        de.create_grid()

                    # Extract all ground truths
                    ground_truth, gt_measurements = extract_all_groud_truths(ground_truth_file)

                    # Create PF
                    w, h = map(int, ground_truth[0][3:5])
                    pf = create_particle_filter(ground_truth[0], q_value, r_value, images_filelist[0], args)

                    # Create images iterator
                    t = create_iterator(images_filelist, args)

                    # Create data association
                    da = MLPFAssociation(states=pf.S, R=r_value, H=pf.H, threshold=args.outlier_detector_threshold)

                    # Iterate of every image
                    features = None
                    for i, im in enumerate(t):
                        img = read_img(im)

                        # Do prediction
                        pf.predict()

                        if i % args.resample_every == 0:

                            # Compute features
                            features = np.array(fd.compute_features(img))

                            if len(features) > 0:
                                # Do data association
                                if args.point_estimate:
                                    features = [
                                        np.array([i[0] + w/2, i[1] + h/2]) for i in features]
                                psi, outlier, c = da.associate(features)

                                pf.update(psi, outlier, c)
                            else:
                                pf.assign_predicted()

                        else:
                            pf.assign_predicted()

                        gt = list(map(int, ground_truth[i]))
                        x = pf.get_x().T

                        # Plot features
                        if args.should_plot:
                            plot_result = plot(args, x, de, img, gt, features)
                            if args.show_plots is False:
                                plotted.append(plot_result)

                        if args.extract_density:
                            if args.should_plot is False:
                                de.estimate(x)
                            if args.point_estimate:
                                v = np.linalg.norm(
                                    [de.xtmean - gt[0] - gt[2]/2, de.ytmean - gt[1] - gt[3]/2], axis=0)
                            else:
                                v = np.linalg.norm([de.xtmean - gt[1],
                                                    de.ytmean - gt[2],
                                                    (de.xbmean - de.xtmean) - gt[3],
                                                    (de.ybmean - de.ytmean) - gt[4]])
                            t.set_description(f"Last error: {v:3.4}")
                            errors.append(v)
                            iou.append(get_iou([de.xtmean, de.ytmean, de.xbmean, de.ybmean],
                                               [gt[1], gt[2], gt[1] + gt[3], gt[2] + gt[4]]))
                            results_dir.append(
                                [de.xtmean, de.ytmean, de.xbmean - de.xtmean, de.ybmean - de.ytmean])
                            break
                        break
                    break
            except Exception as e:
                print(f"Crashed with error: {str(e)}")
            finally:
                with open(f"results/pf_box_R_{r_ind}_Q_{q_ind}_{args.object_detector}.pickle", 'wb') as fp:
                    pickle.dump(results, fp)
