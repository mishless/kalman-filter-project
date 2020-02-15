import argparse
import cv2
from yolo.yolo_object_detection import YOLOObjectDetection
from glob import glob
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Run YOLO on input folder and show result in popup window")
parser.add_argument("input_dataset", type=str,
                    help="Folder with images where to run YOLO on")
args = parser.parse_args()

dataset = args.input_dataset
ground_truth_file = dataset + "/groundtruth_rect.txt"
images_wildcard = dataset + "/img/*.jpg"
images_filelist = glob(images_wildcard)

yolo_object = YOLOObjectDetection()

for image in images_filelist:
    img = cv2.imread(image)
    image_height, image_width, channels = img.shape
    features = yolo_object.compute_features(img)

    for feature in features:
        cv2.rectangle(img, (int(feature[0]), int(feature[1])), (int(feature[2]), int(feature[3])), (0, 255, 0), 2)

    plt.gca()
    plt.cla()
    plt.imshow(img)
    plt.gca().autoscale(False)
    plt.pause(0.1)
