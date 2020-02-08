import argparse
import cv2
import numpy as np
from ssd.ssd import SSDObjectDetection
from glob import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Run SSD on input folder and show result in popup window")
parser.add_argument("input_dataset", type=str,
                    help="Folder with images where to run SSD on")
args = parser.parse_args()

dataset = args.input_dataset
ground_truth_file = dataset + "/groundtruth_rect.1.txt"
images_wildcard = dataset + "/img/*.jpg"
images_filelist = glob(images_wildcard)

frozen = "ssd/frozen_inference_graph.pb"
pbtxt = "ssd/ssd_inception_v2_coco_2017_11_17.pbtxt"
ssd_object = SSDObjectDetection(frozen, pbtxt)
plt.ion()

for image in images_filelist:
    img = cv2.imread(image)
    image_height, image_width, channels = img.shape
    outs = ssd_object.compute_features(img)[0]

    for detection in outs[0, 0, :, :]:
        # get the class index
        class_index = int(detection[1])
        # get the score
        score = float(detection[2])
        # draw the bounding box
        threshold = 0.2
        if score > threshold:
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            cv2.rectangle(img, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 255, 0), 2)

    plt.gca()
    plt.cla()
    plt.imshow(img)
    plt.gca().autoscale(False)
    plt.pause(0.1)

