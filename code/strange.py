# A module of different strangeness functions for Part I of the anomaly 
# detection process.  
# General Form:
#   INPUT: a set of training examples
#   OUTPUT: a set of corresponding alpha values

from __future__ import division
import numpy as np


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
    norms = [np.linalg.norm(i) for i in train_preds]
    min_val = np.min(norms)
    max_val = np.max(norms)
    range_percetile = [(i - min_val) / (max_val - min_val) for i in norms]
    return range_percetile
