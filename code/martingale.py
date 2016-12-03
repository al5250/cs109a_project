# A module of different Martingale creation functions for Part II of the  
# anomaly detection process.  
# General Form:
#   INPUT: a set of p values
#   OUTPUT: a set of corresponding Martingales

from __future__ import division
import numpy as np
import random 
import betting


def power(p_vals, epsilon=0.5):
    mgales = []
    acc = 1
    for p in p_vals:
        acc *= betting.fixed(p, epsilon)
        mgales.append(acc)
    return mgales


def simple_mixture(p_vals, n_points=10):
    mgales = []
    epsilons = np.linspace(0, 1, n_points)
    h = epsilons[1] - epsilons[0]
    accs = np.ones(n_points)
    for p in p_vals:
        accs = [betting.fixed(p, epsilon) * acc for epsilon, acc in zip(epsilons, accs)]
        mgales.append(h * np.trapz(accs))
    return mgales
