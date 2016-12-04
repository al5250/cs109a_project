# A module of different strangeness functions for Part I of the anomaly
# detection process.
# General Form:
#   INPUT: a set of training examples
#   OUTPUT: a set of corresponding alpha values

from __future__ import division
import numpy as np
from sklearn import linear_model
import math

def avg_distance(train_preds):
    n = len(train_preds)
    avg_distances = []
    for i in train_preds:
        distances_to_i = []
        for j in train_preds:
            distances_to_i.append(np.linalg.norm(i - j))
        avg_distances.append(1 / n * sum(distances_to_i))
    return avg_distances


def range_percentile(train_preds):
    if len(train_preds) == 1:
        return [1]
    norms = [np.linalg.norm(i) for i in train_preds]
    min_val = np.min(norms)
    max_val = np.max(norms)
    range_percentile = [(i - min_val) / (max_val - min_val) for i in norms]
    return range_percentile


def ols_trend(train_preds):
    n = len(train_preds)
    window_size = max(2, n//4) # TODO: clean window size up
    timestamps = np.arange(n).reshape(-1, 1)
    train_preds = train_preds.reshape(-1, 1)
    ols_trends = []
    coeffs = []
    lin = linear_model.LinearRegression()
    for i in range(n):
        # if i != 0:
        #     ols_trends.append(abs(train_preds[i] - lin.predict(i)))
        # else:
        #     ols_trends.append(0)
        left = max(0, i - int(math.floor(window_size / 2)))
        right = min(n, i + int(math.ceil(window_size / 2)))
        lin.fit(timestamps[left:right], train_preds[left:right])
        coeffs.append(lin.coef_[0][0])
        if i == 0:
            ols_trends.append(np.linalg.norm(coeffs[i]))
        else:
            ols_trends.append(np.linalg.norm(coeffs[i] - coeffs[i-1]))
    return ols_trends
