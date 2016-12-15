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


def ols_residual(train_preds):
    n = len(train_preds)
    window_size = 10
    timestamps = np.arange(n).reshape(-1, 1)
    train_preds = train_preds.reshape(-1, 1)
    ols_residuals = []
    lin = linear_model.LinearRegression()
    for right in range(n):
        if right == 0:
            ols_residuals.append(0)
        else:
            left = max(0, right - window_size)
            lin.fit(timestamps[left:right], train_preds[left:right])
            ols_residuals.append(abs(lin.predict(right) - train_preds[right]))
    return ols_residuals


def ols_trend(train_preds):
    n = len(train_preds)
    window_size = 10
    timestamps = np.arange(n).reshape(-1, 1)
    train_preds = train_preds.reshape(-1, 1)
    ols_trends = []
    coeffs = []
    lin = linear_model.LinearRegression()
    for right in range(n):
        if right == 0:
            ols_trends.append(0)
            coeffs.append(0)
        else:
            left = max(0, right - window_size)
            lin.fit(timestamps[left:right], train_preds[left:right])
            coeffs.append(lin.coef_[0][0])
            distances = []
            for i in range(right):
                distances.append(abs(coeffs[right] - coeffs[i]))
            ols_trends.append(np.mean(distances))
    return ols_trends
