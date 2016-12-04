# A module of different betting functions for Part II of the anomaly
# detection process.
# General Form:
#   INPUT: a p value (and necessary parameters)
#   OUTPUT: a representative value for Martingale calculation

from __future__ import division
import numpy as np
from sklearn.neighbors.kde import KernelDensity


def fixed(p, epsilon):
    if epsilon < 0 or epsilon > 1:
        raise ValueError('Epsilon value out of bounds.')
    return epsilon * p ** (epsilon - 1)

def plugin(p_vals, kernel, n_points):
    p_vals = np.array(p_vals).reshape(-1, 1)
    p_vals_extended = np.vstack([p_vals, -p_vals, 2-p_vals])
    kde = KernelDensity(kernel=kernel).fit(p_vals_extended)
    x = np.linspace(0, 1, n_points)
    y = np.exp([kde.score(pt) for pt in x])
    norm_factor = np.trapz(y, x)
    self_prob = np.exp(kde.score([p_vals[-1]]))
    return self_prob / norm_factor
