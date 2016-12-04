# A class for anomaly detection.  There are two main customizable options:
#      1) A strangeness function selected from the 'strange' module.
#      2) A Martingale creation function selected from the 'martingale' module.

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import strange
import martingale

class AnomalyDetector(object):

    # Initialization
    def __init__(self, strange_func, mgale_type, mgale_params={}, threshold=20):
        self.strange_func = strange_func
        self.mgale_type = mgale_type
        self.mgale_params = mgale_params
        self.threshold = threshold

    # Part I: Generates p values from training examples.
    def get_p_vals(self, train_preds):
        n = len(train_preds)
        strange_dict = {
            'AverageDistance': strange.avg_distance,
            'RangePercentile': strange.range_percentile,
            'OLSTrend': strange.ols_trend
        }
        p_vals = []
        for i in range(0, n):
            subset = train_preds[:(i+1)]
            alphas = strange_dict[self.strange_func](subset)
            greater_count = np.sum([alph > alphas[i] for alph in alphas])
            equal_count = np.sum([alph == alphas[i] for alph in alphas])
            theta = random.uniform(0, 1)
            p_vals.append((greater_count + theta * equal_count) / (i + 1))
        return p_vals

    # Part II: Generates Martingales from p values.
    def get_mgales(self, p_vals):
        mgale_dict = {
            'Power': martingale.power,
            'SimpleMixture': martingale.simple_mixture,
            'Plugin': martingale.plugin
        }
        params = self.mgale_params
        params['p_vals'] = p_vals
        mgales = mgale_dict[self.mgale_type](**params)
        return mgales

    # Plots Martingales values, highlighting the ones that cross the threshold.
    def plot_mgales(self, mgales):
        n = len(mgales)
        plt.figure()
        for i in range(0, n):
            if mgales[i] < self.threshold:
                plt.plot(i, mgales[i], 'bo')
            else:
                plt.plot(i, mgales[i], 'ro')
        plt.plot(range(0, n), mgales, 'k-')
        plt.plot([0, n], self.threshold * np.ones(2), 'r-')
        plt.xlabel('Training Examples')
        plt.ylabel('Martingale Values')
        plt.show()

    # Combines get_p_values, get_mgales, and plot_mgales into one function.
    def analyze(self, train_preds):
        p_vals = self.get_p_vals(train_preds)
        mgales = self.get_mgales(p_vals)
        self.plot_mgales(mgales)
