import cv2

class SSDObjectDetection:
    def __init__(self, frozen, pbtxt):
        self.net = cv2.dnn.readNetFromTensorflow(frozen, pbtxt)
        layer_names = self.net.getLayerNames()

        self.outputLayers = [layer_names[i[0] - 1]
                             for i in self.net.getUnconnectedOutLayers()]

    def compute_outputs(self, img):
        # Image scaling and R-B channel swapping
        blob = cv2.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB = True, crop = False)
        self.net.setInput(blob)

        # Outputs
        outs = self.net.forward(self.outputLayers)
        return outs

    def compute_features(self, img):
        outs = self.compute_outputs(img)

        image_height, image_width, _ = img.shape
        features = []
        for out in outs:
            for detection in out[0, 0, :, :]:
                class_index = int(detection[1])
                score = float(detection[2])
                threshold = 0.5
                if score > threshold:
                    left = detection[3] * image_width
                    top = detection[4] * image_height
                    right = detection[5] * image_width
                    bottom = detection[6] * image_height
                    features.append([int(left), int(top), int(right), int(bottom)])
        return features
