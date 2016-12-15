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
            'OLSTrend': strange.ols_trend,
            'OLSResidual': strange.ols_residual
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

    # Plots Martingales values, highlighting the change points given.
    def plot_mgales(self, ax, mgales, train_preds, anomalies):
        n = len(mgales)
        ax.plot(range(n), np.log10(mgales), 'b')
        for a in anomalies:
            ax.axvline(x=a, color='r', linestyle='--')
        ax.set_xlabel('Time')
        ax.set_ylabel('Martingale Values (log)')
        ax.set_title('{} {} Martingale Values'.format(
            self.strange_func, self.mgale_type))

        return ax

    # Combines get_p_values, get_mgales, and plot_mgales into one function.
    def analyze(self, ax, train_preds, anomalies):
        p_vals = self.get_p_vals(train_preds)
        mgales = self.get_mgales(p_vals)
        change_detected = max(mgales) > self.threshold
        return change_detected, self.plot_mgales(ax, mgales, train_preds, anomalies)
