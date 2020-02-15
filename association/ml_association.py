from math import pi

import numpy as np


class MLKalmanAssociation:
    def __init__(self, R, H, threshold):
        self.mu_bar = None
        self.Sigma_bar = None
        self.R = R
        self.H = H
        self.threshold = threshold

    def update_prediction(self, mu_bar, Sigma_bar):
        self.mu_bar = mu_bar
        self.Sigma_bar = Sigma_bar

    def associate(self, measurements):
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
        centers = list(map(lambda m: np.array([m[0] + (m[2] - m[0])/2, m[1] + (m[3] - m[1])/2]), measurements))
        for i, (mx, my) in enumerate(centers):
            new_z = np.array([mx - self.mu_bar[0], my -
                              self.mu_bar[2]]).reshape(2, 1)
            nu[i] = np.ravel(new_z)
            p = ctn * np.exp(-0.5 * new_z.T.dot(Sinv).dot(new_z))
            phi[i] = p
        if len(phi) > 0:
            c = np.argmax(phi)
            mah = nu[c].T.dot(Sinv).dot(nu[c])
            if mah < self.threshold:
                return centers[c]
        return None


class MLPFAssociation:
    def __init__(self, states, R, H, threshold):
        self.S_bar = states
        self.R = R
        self.Rinv = np.linalg.pinv(R)
        self.eta = 1.0 / ((2 * pi * np.linalg.det(R)) ** 0.5)
        self.threshold = threshold
        self.M = len(states)
        self.H = H

    def associate(self, measurements):
        """Associate one of the measurements (objects detected) to the predicted
        position.

        Do an outlier check if the mahalonobis distance is higher than the
        threshold. If it is, return the feature. Else, return None.

        :param measurements: Array of measured features
        :return position of the measured object (x, y) or None
        """
        n = len(measurements)
        exps = np.zeros((self.M, n))
        Psi = np.zeros((self.M, n))
        c = np.zeros((self.M, n))

        # Observation model
        z_hat = self.observation_model()

        # Difference
        nu = z_hat.reshape(-1, 1, 2) - measurements.reshape(1, -1, 2)

        # Calculate Mahalonobis distance
        for i in range(n):
            for m in range(self.M):
                cur_nu = nu[m, i]
                exps[m, i] = cur_nu.reshape(1, -1).dot(self.Rinv).dot(cur_nu.reshape(-1, 1))

        psi = self.eta * np.exp(-0.5 * exps)

        outlier = psi.max(axis=0) < self.threshold

        c = np.argmax(psi, axis=1)

        return psi, outlier, c

    def observation_model(self):
        z_hat = self.S_bar.dot(self.H.T)
        return z_hat
