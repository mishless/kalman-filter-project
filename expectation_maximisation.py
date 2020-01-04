import numpy as np
from scipy.stats import multivariate_normal

from kalman_filter import KalmanFilter


class EM:

    def __init__(self):
        self.kf = KalmanFilter()

    def find_Q(self, data=None):
        log_likelihoods = []
        for i in range(0, 20):
            x_posteriors, x_predictions, cov = self.run(data=data)
            log_likelihoods.append(self.calculate_log_likelihood(x_posteriors, cov, data))
            self.m_step(x_predictions, x_posteriors, data)
        return log_likelihoods

    def run(self, data=None):
        x_posteriors, x_predictions, cov = [], [], []
        for z in data:
            self.kf.predict()
            x_predictions.append(self.kf.x)
            self.kf.update(z)
            x_posteriors.append(self.kf.x)
            cov.append(self.kf.P)
        x_posteriors, x_predictions, cov = np.array(x_posteriors), np.array(x_predictions), np.array(cov)
        return x_posteriors, x_predictions, cov

    def calculate_log_likelihood(self, x_posteriors, cov, measurements):
        log_likelihood = 0
        for i in range(0, len(cov)):
            S = np.dot(np.dot(self.kf.H, cov[i]), self.kf.H.T) + self.kf.R
            state_posterior_in_meas_space = np.dot(self.kf.H, x_posteriors[i]).squeeze()
            distribution = multivariate_normal(mean=state_posterior_in_meas_space, cov=S)
            log_likelihood += np.log(distribution.pdf(measurements[i]))
        return log_likelihood

    def m_step(self, x_predictions, x_posteriors, measurements):
        self.kf = KalmanFilter()
        self.kf.Q = np.cov((x_posteriors - x_predictions).squeeze().T, bias=True)
        self.kf.R = np.cov((measurements.T - np.dot(self.kf.H, x_posteriors.squeeze().T)), bias=True)
