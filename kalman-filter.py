import numpy as np

class KalmanFilter:
    def __init__(self):
        self.A = np.array([[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.R = np.array([[1, 0],
                           [0, 1]])
        self.Q = np.array([[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 1, 1]])
        self.P = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.u = 0
        self.x = np.array([[0], [1], [0], [1]])
        self.cov = self.P

        self.x_predicted = self.x
        self.cov_predicted = self.cov

    def predict(self):
        self.x_predicted = self.A * self.x + self.B * self.u
        self.cov_predicted = self.A * self.cov * self.A.T + self.Q

    def update(self, measurement):
        S = self.H * self.cov_predicted * self.H.T + self.R
        y = measurement - self.H * self.x_predicted
        K = self.cov_predicted * self.H.T * np.linalg.pinv(S)
        self.x = self.x_predicted + K * y
        self.cov = (np.identity(self.P.shape[0]) - K * self.H) * self.cov_predicted
