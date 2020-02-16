from random import uniform

import numpy as np


class ParticleFilter:
    def __init__(self, num_particles, Q=None, R=None, img_shape=(100, 100),
                 mean_speed=10, resample_mode="systematic",
                 initial_state=None, point_estimate=True):
        self.M = num_particles
        self.point_estimate = point_estimate
        num_components = 4 if point_estimate else 8
        self.S = np.empty((num_particles, num_components))

        # Initialize states
        self.initialize_states(img_shape, mean_speed,
                               initial_state, point_estimate)

        self.S_bar = np.copy(self.S)
        self.weights = np.ones(num_particles) / num_particles

        # Model
        if point_estimate:
            self.A = np.array([[1, 1, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]])
        else:
            self.A = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 1],
                               [0, 0, 0, 0, 0, 0, 0, 1]])
        if Q is not None:
            self.Q = Q
        else:
            if point_estimate:
                self.Q = np.array([[1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [0, 0, 1, 1],
                                   [0, 0, 1, 1]])
            else:
                self.Q = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                   [1, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 1],
                                   [0, 0, 0, 0, 0, 0, 1, 1]])

        if R is not None:
            self.R = np.eye(num_components//2) * R
        else:
            self.R = np.eye(num_components//2)

        # Control
        self.B = 0
        self.u = 0

        # Observation model
        if point_estimate:
            self.H = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0]])
        else:
            self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0]])

        self.resample_mode = resample_mode
        self.img_shape = img_shape

    def update(self, psi, outlier, c):
        if not np.any(outlier):
            # Reweight
            self.reweight(psi, outlier, c)

            # Resample
            if self.resample_mode is None:
                self.S = self.S_bar
            elif self.resample_mode == "multinomial":
                self.multinomal_resample()
            elif self.resample_mode == "systematic":
                self.systematic_resample()
            else:
                raise NotImplementedError
        else:
            self.assign_predicted()

    def predict(self):
        x = np.dot(self.S, self.A.T) + np.dot(self.u, self.B)
        x = x + np.random.randn(self.M, len(self.Q)).dot(self.Q ** 0.5)
        self.S_bar[:] = x

    def reweight(self, psi, outlier, c):
        n = len(outlier)
        for i in range(n):
            if not outlier[i]:
                mask = c == i
                self.weights[mask] *= psi[mask, i]
        self.weights /= self.weights.sum()

    def initialize_states(self, img_shape, max_speed, initial_state, point_estimate):
        if point_estimate:
            if initial_state is not None:
                # Global tracking
                s = np.random.randn(len(self.S), 2) * 25
                self.S[:, 0] = s[:, 0] + initial_state[0]  # x
                self.S[:, 2] = s[:, 1] + initial_state[1]  # y
            else:
                # Local tracking
                s = np.random.uniform(0, 1, size=(len(self.S), 2))
                self.S[:, 0] = s[:, 0] * img_shape[1]  # x
                self.S[:, 2] = s[:, 1] * img_shape[0]  # y

            self.S[:, 1] = max_speed * np.random.randn(len(self.S))  # vx
            self.S[:, 3] = max_speed * np.random.randn(len(self.S))  # vy
        else:
            if initial_state is not None:
                # Global tracking
                s = np.random.randn(len(self.S), 4) * 25
                self.S[:, 0] = s[:, 0] + initial_state[0]  # t-x
                self.S[:, 2] = s[:, 1] + initial_state[1]  # t-y
                self.S[:, 4] = s[:, 1] + initial_state[2]  # b-x
                self.S[:, 6] = s[:, 1] + initial_state[3]  # b-y
            else:
                # Local tracking
                s = np.random.uniform(0, 1, size=(len(self.S), 4))
                self.S[:, 0] = s[:, 0] * img_shape[1]  # t-x
                self.S[:, 2] = s[:, 1] * img_shape[0]  # t-y
                self.S[:, 4] = s[:, 2] * img_shape[1]  # b-x
                self.S[:, 6] = s[:, 3] * img_shape[0]  # b-y

            self.S[:, [1, 3, 5, 7]] = max_speed * \
                np.random.randn(len(self.S), 4)

    def systematic_resample(self):
        cdf = np.cumsum(self.weights)
        w = 1 / self.M
        r = uniform(0, w)
        for m in range(self.M):
            i = np.argmax(cdf >= (r + (m - 1) * w))
            self.S[m] = self.S_bar[i]
            self.weights[m] = w

    def multinomal_resample(self):
        cdf = np.cumsum(self.weights)
        w = 1 / self.M
        for m in range(self.M):
            r = uniform(0, 1)
            i = np.argmax(cdf >= r)
            self.S[m] = self.S_bar[i]
            self.weights[m] = w

    def get_x(self):
        if self.point_estimate:
            return self.S[:, [0, 2]]
        else:
            return self.S[:, [0, 2, 4, 6]]

    def assign_predicted(self):
        self.S[:] = self.S_bar
