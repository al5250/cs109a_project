# A module of different betting functions for Part II of the anomaly 
# detection process.  
# General Form:
#   INPUT: a p value (and necessary parameters)
#   OUTPUT: a representative value for Martingale calculation

from __future__ import division


def fixed(p, epsilon):
    if epsilon < 0 or epsilon > 1:
        ValueError('Epsilon value out of bounds.')
    return epsilon * p ** (epsilon - 1)