# A script file to compare the performance of various thresholds
# in detecting anomalous datasets. Plots accuracy vs. threshold values
# for anomalous and non-anomalous data.


from __future__ import division
import numpy as np
from anomaly_detector import AnomalyDetector
import gen_data
import matplotlib.pyplot as plt
import time

# generate data
n_datasets = 50
anomaly_types = [
    ('Normal', None),
    ('SlopeChange', 'OLSTrend'),
    ('RandomOutliers', 'OLSResidual'),
    ('SuddenGap', 'AverageDistance')
]
datasets = {
    anomaly[0]: [gen_data.gen_data(anomaly[0])[0] for i in range(n_datasets)]
    for anomaly in anomaly_types
}

# calculate accuracies
thresholds = np.arange(0.5, 1.6, 0.1)
accuracies = dict()

start_time = time.time()
for anomaly, strange_func in anomaly_types:
    if anomaly == 'Normal':
        continue
    print "Processing data for anomaly type {}".format(anomaly)
    accuracy = []
    anomalous_maxes = []
    normal_maxes = []

    det = AnomalyDetector(strange_func, 'Power', mgale_params={'epsilon': 0.8})

    print "Getting martingale values for non-anomalous data..."
    start = time.time()
    for i, normal_stream in enumerate(datasets['Normal']):
        if i != 0 and i % 10 == 0:
            print "{} / {} done in {:.3} minutes".format(i,
            len(datasets[anomaly]), (time.time() - start) / 60)
        p_vals = det.get_p_vals(normal_stream)
        mgales = det.get_mgales(p_vals)
        normal_maxes.append(max(mgales))

    print "Getting martingale values for anomalous data..."
    start = time.time()
    for i, anomalous_stream in enumerate(datasets[anomaly]):
        if i != 0 and i % 10 == 0:
            print "{} / {} done in {:.3} minutes".format(i,
            len(datasets[anomaly]), (time.time() - start) / 60)
        p_vals = det.get_p_vals(anomalous_stream)
        mgales = det.get_mgales(p_vals)
        anomalous_maxes.append(max(mgales))

    for t in 10 ** thresholds:
        change_detected_norm = [m > t for m in normal_maxes]
        change_detected_anom = [m > t for m in anomalous_maxes]
        accuracy.append( (np.mean(change_detected_anom),
            1 - np.mean(change_detected_norm)) )

    accuracies[anomaly] = accuracy

    # plot accuracies vs threshold values
    anom_acc, norm_acc = zip(*accuracy)
    plt.figure()
    plt.plot(thresholds, anom_acc, c='r', label='Anomalous Data')
    plt.scatter(thresholds, anom_acc, c='r')
    plt.plot(thresholds, norm_acc, c='g', label='Non-anomalous Data')
    plt.scatter(thresholds, norm_acc, c='g')
    plt.xlabel('Threshold Values (log)')
    plt.ylabel('Detection Accuracies')
    plt.title('{} Data'.format(anomaly))
    plt.legend(loc='best')

print "Process completed in {:.3} minutes".format((time.time() - start_time) / 60)
plt.show()
