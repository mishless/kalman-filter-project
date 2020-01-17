import numpy as np
import matplotlib.pyplot as plt


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
        ctn = np.linalg.det(2 * np.pi * S)**(-0.5)
        nu = np.zeros((n, 2))
        plt.cla()
        for i, (mx, my) in enumerate(measurements):
            plt.imshow(plt.imread(img))
            plt.plot(self.mu_bar[0], self.mu_bar[2], marker='o', color='blue')
            plt.plot(mx, my, marker='o', color='lightgreen')
            plt.show()
            new_z = np.array([mx - self.mu_bar[0], my -
                              self.mu_bar[2]]).reshape(2, 1)
            nu[i] = np.ravel(new_z)
            p = ctn * np.exp(-0.5 * new_z.T.dot(Sinv).dot(new_z))
            phi[i] = p
            plt.cla()
        c = np.argmax(phi)
        mah = nu[c].T.dot(Sinv).dot(nu[c])
        print(mah)
        if mah > self.threshold:
            return nu[c]
        else:
            return None