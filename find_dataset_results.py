import numpy as np
import pickle
with open(f"results/datasets_iou.pickle", 'rb') as f:
    results = pickle.load(f)
means = {}
for key, value in results.items():
    means[key] = [np.mean(value), value.count(0)]

result = {k: v for k, v in sorted(means.items(), key=lambda item: item[1][0])}
print(result)
with open(f"results/datasets_iou_full_yolo.pickle", 'rb') as f:
    results = pickle.load(f)
means = {}
for key, value in results.items():
    means[key] = [np.mean(value), value.count(0)]

result = {k: v for k, v in sorted(means.items(), key=lambda item: item[1])}
print(result)
