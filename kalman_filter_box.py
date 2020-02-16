import numpy as np


class KalmanFilter:
    def __init__(self, x=None, R=None, Q=None, P=None, cov=None):
        self.A = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # Initialize the process covariance
        if R is not None:
            self.R = np.eye(4) * R
        else:
            self.R = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        self.Q = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                           [1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 1],
                           [0, 0, 0, 0, 0, 0, 1, 1]])
        if Q is not None:
            self.Q = self.Q * Q
        self.P = np.diag(1 * np.ones(8))
        self.u = 0
        if x is not None:
            self.x = x
        else:
            self.x = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
        self.cov = self.P

        self.x_predicted = self.x
        self.cov_predicted = self.cov

    def predict(self):
        self.x_predicted = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.cov_predicted = np.dot(np.dot(self.A, self.cov), self.A.T) + self.Q
        return self.x_predicted, self.cov_predicted

    def update(self, measurement):
        S = np.dot(np.dot(self.H, self.cov_predicted), self.H.T) + self.R
        if measurement is None:
            self.x = self.x_predicted
            self.cov = self.cov_predicted
        else:
            m = measurement.reshape(1, 4)
            y = m.T - np.dot(self.H, self.x_predicted)
            K = np.dot(np.dot(self.cov_predicted, self.H.T), np.linalg.pinv(S))
            self.x = self.x_predicted + np.dot(K, y)
            self.cov = np.dot(
                (np.identity(self.P.shape[0]) - np.dot(K, self.H)), self.cov_predicted)

    def get_x(self):
        return self.x[0][0], self.x[2][0], self.x[4][0], self.x[6][0]
