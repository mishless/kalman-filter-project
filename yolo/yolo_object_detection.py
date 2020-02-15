import cv2
import numpy as np


class YOLOObjectDetection:
    def __init__(self, type):
        # Load YOLO V3
        if type == 'full':
            self.net = cv2.dnn.readNet("yolo/yolov3.weights",
                                       "yolo/yolov3.cfg")
        elif type == 'tiny':
            self.net = cv2.dnn.readNet("yolo/tiny/yolov3.weights",
                                       "yolo/tiny/yolov3.cfg")
        layer_names = self.net.getLayerNames()

        # Define the output layers
        self.outputLayers = [layer_names[i[0] - 1]
                             for i in self.net.getUnconnectedOutLayers()]

    def compute_features(self, img):
        # Features list
        features = []
        # Load input image
        height, width, channels = img.shape
        # Image scaling and R-B channel swapping
        blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=(416, 416),
                                     mean=(0, 0, 0), swapRB=True, crop=False)

        # YOLO V3 input
        self.net.setInput(blob)
        # Outputs
        outs = self.net.forward(self.outputLayers)
        confidences = []
        boxes = []

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
                    #features.append(np.array([center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]))
                    confidences.append(float(confidence))
                    boxes.append([center_x - w/2, center_y - h/2, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            features.append(np.array([left, top, left + width, top + height]))

        return features

    def compute_features_for_box(self, image_filename):
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
        for i0, out in enumerate(outs):
            for i1, detection in enumerate(out):
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    features.append(np.array([center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]))
        return features
