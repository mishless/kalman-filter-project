import cv2
import numpy as np


class YOLOObjectDetection:
    def __init__(self, confidence=0.5):
        self.confidence = confidence

        # Load YOLO V3
        self.net = cv2.dnn.readNet("yolo/yolov3.weights",
                                   "yolo/yolov3.cfg")
        layer_names = self.net.getLayerNames()

        # Define the output layers
        self.outputLayers = [layer_names[i[0] - 1]
                             for i in self.net.getUnconnectedOutLayers()]

    def compute_features(self, image_filename):
        # Features list
        features = []

        # Load input image
        img = cv2.imread(image_filename)
        height, width, channels = img.shape

        # Image scaling and R-B channel swapping
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(416, 416),
                                     mean=(0, 0, 0), swapRB=True, crop=False)

        # YOLO V3 input
        self.net.setInput(blob)

        # Outputs
        outs = self.net.forward(self.outputLayers)

        # Parse YOLO results. Draw circle and rectangle
        # Output: (xt, yt, xb, yb)
        for i0, out in enumerate(outs):
            for i1, detection in enumerate(out):
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    width = int(detection[2] * width)
                    height = int(detection[3] * height)
                    features.append(np.array(
                        [center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2]).astype(int))

        return features
