from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def gen_data(anomaly_type='Normal', length=150, n_change_points=2):
    """
    anomaly_type: string. `RandomOutliers` | `SuddenGap` | `SlopeChange`
    length: integer. length of time series
    n_change_points: integer. number of distribution change points
    returns
        time series: numpy array of shape (length, )
        anomalies: numpy array of shape (n_change_points, )
    """
    segments = []
    anomalies = []

    anomaly_length = length // (2 * (2* n_change_points + 1))
    normal_length = (length - n_change_points * anomaly_length) // (n_change_points+1)
    seg_length = anomaly_length + normal_length

    if anomaly_type == 'SlopeChange':
        m = 5 * np.random.uniform(0, 1)
        last_m = last_b = 0
        for i in range(n_change_points):
            m1 = np.random.uniform(10, 15)
            x0 = i * (normal_length + anomaly_length)
            y0 = last_m * x0 + last_b
            b0 = y0 - m * x0
            b1 = (m - m1) * (x0 + normal_length) + b0

            anomalies.append(x0 + normal_length)
            segments.append(m * np.arange(x0, x0 + normal_length) + b0 +
                np.random.normal(0, 1, normal_length))
            segments.append(m1 * np.arange(x0 + normal_length, x0 + seg_length) + b1 +
                np.random.normal(0, 1, anomaly_length))

            last_m = m1
            last_b = b1

    elif anomaly_type == 'RandomOutliers' or anomaly_type == 'SuddenGap':
        peak = np.random.uniform(10, 15)
        if anomaly_type == 'RandomOutliers':
            anomaly_length = 3
        for i in range(n_change_points):
            x0 = i * (normal_length + anomaly_length)
            anomalies.append(x0 + normal_length)
            segments.append(np.random.normal(0, 1, normal_length))
            segments.append(peak + np.random.normal(0, 1, anomaly_length))
        segments.append(np.random.normal(0, 1, normal_length))

    elif anomaly_type == 'Normal':
        segments.append(np.random.normal(0, 1, length))

    return np.hstack(segments), anomalies


def plot_data(ax, data, anomalies):
    n = len(data)
    # ax.plot(range(n), data, 'bo')
    ax.plot(range(n), data, 'b')
    for a in anomalies:
        ax.axvline(x=a, color='r', linestyle='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Time Series Values')
    ax.set_title('Original Time Series Data')
    return ax
