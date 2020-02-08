import cv2


class SSDObjectDetection:
    def __init__(self, frozen, pbtxt):
        self.net = cv2.dnn.readNetFromTensorflow(frozen, pbtxt)
        layer_names = self.net.getLayerNames()

        self.outputLayers = [layer_names[i[0] - 1]
                             for i in self.net.getUnconnectedOutLayers()]

    def compute_features(self, img):
        # Image scaling and R-B channel swapping
        blob = cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True)
        self.net.setInput(blob)

        # Outputs
        outs = self.net.forward(self.outputLayers)
        return outs
