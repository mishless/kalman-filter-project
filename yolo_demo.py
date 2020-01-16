import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(
    description="Run YOLO V3 on input image and show result in popup window")
parser.add_argument("input_image", type=str,
                    help="Image where to run YOLO V3 on")
args = parser.parse_args()

# Load YOLO V3
net = cv2.dnn.readNet("yolo/yolov3.weights",
                      "yolo/yolov3.cfg")
layer_names = net.getLayerNames()

# Define the output layers
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load input image
img = cv2.imread(args.input_image)
height, width, channels = img.shape

# Image scaling and R-B channel swapping
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416),
                             mean=(0, 0, 0), swapRB=True, crop=False)

# YOLO V3 input
net.setInput(blob)

# Outputs
outs = net.forward(outputLayers)

# Parse YOLO results. Draw circle and rectangle
for i0, out in enumerate(outs):
    for i1, detection in enumerate(out):
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Plot image with rectangle and circle
cv2.imshow("bow", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
