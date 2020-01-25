import matplotlib.pyplot as plt
import numpy as np


class MLAssociation:
    def __init__(self, R, H, threshold):
        self.mu_bar = None
        self.Sigma_bar = None
        self.R = R
        self.H = H
        self.threshold = threshold

    def update_prediction(self, mu_bar, Sigma_bar):
        self.mu_bar = mu_bar
        self.Sigma_bar = Sigma_bar

    def associate(self, img, measurements):
        """Associate one of the measurements (objects detected) to the predicted
        position.

        Do an outlier check if the mahalonobis distance is higher than the
        threshold. If it is, return the feature. Else, return None.

        :param measurements: Array of measured features
        :return position of the measured object (x, y) or None
        """
        n = len(measurements)
        phi = np.zeros(n)
        S = self.H.dot(self.Sigma_bar).dot(self.H.T) + self.R
        Sinv = np.linalg.pinv(S)
        ctn = np.linalg.det(2 * np.pi * S) ** (-0.5)
        nu = np.zeros((n, 2))
        for i, (mx, my) in enumerate(measurements):
            new_z = np.array([mx - self.mu_bar[0], my -
                              self.mu_bar[2]]).reshape(2, 1)
            nu[i] = np.ravel(new_z)
            p = ctn * np.exp(-0.5 * new_z.T.dot(Sinv).dot(new_z))
            phi[i] = p
        c = np.argmax(phi)
        plt.figure()
        plt.imshow(plt.imread(img))
        plt.plot(self.mu_bar[0], self.mu_bar[2], marker='o', color='blue')
        plt.plot(measurements[c, 0], measurements[c, 1], marker='o', color='lightgreen')
        plt.show()
        mah = nu[c].T.dot(Sinv).dot(nu[c])
        if mah < self.threshold:
            return measurements[c]
        else:
            return None
