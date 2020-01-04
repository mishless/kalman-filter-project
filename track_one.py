import csv
from kalman_filter import KalmanFilter
from expectation_maximisation import EM
import numpy as np
import matplotlib.pyplot as plt

images = []


def main():
    diff = 0
    measurements = []
    f = open("data/basketball/groundtruth_rect.txt")
    for row in csv.reader(f):
        measurements.append(np.array([int(int(row[0]) - int(row[2]) / 2), int(int(row[1]) - int(row[3]) / 2)]))

    # Find Q using EM
    em_obj = EM()
    data = np.array(measurements)
    likelihoods = em_obj.find_Q(data)
    print(em_obj.kf.Q)
    plt.plot(likelihoods)
    plt.show()

    kf = KalmanFilter(np.array([[measurements[0][0]], [1], [measurements[0][1]], [1]]))
    kf.Q = em_obj.kf.Q
    for m in measurements[1:]:
        kf.predict()
        kf.update(m)
        print(f"Predited position: {kf.x[0][0], kf.x[2][0]}, Ground truth position: {m[0], m[1]}")
        diff += np.linalg.norm([kf.x[0][0], kf.x[2][0]] - m, axis=0)
    print(diff/len(measurements))


if __name__ == '__main__':
    main()
